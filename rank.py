#!/usr/bin/env python3
"""
rank.py
========
Single-command entrypoint that produces the submission CSV from the
candidates file, exactly as required by submission_spec.md section 10.3:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline
---------
1. Load and parse the job description into structured requirements
   (jd_parser.py) plus a semantic-text query.
2. Stream-load all candidates from the JSONL file.
3. Fit a TF-IDF + SVD semantic space jointly over the JD text and every
   candidate's full free-text profile (semantic_matcher.py), then compute
   each candidate's cosine similarity to the JD.
4. Score every candidate (scorer.py): structured feature checks,
   multiplicative disqualifiers, behavioral-signal multiplier, honeypot
   hard gate.
5. Take the top 100 by score, generate fact-grounded reasoning strings
   (reasoning_generator.py), and write the CSV in the exact required
   format (header, rank 1-100, non-increasing score, deterministic
   candidate_id tiebreak).

Compute budget
---------------
Designed to comfortably clear the challenge's 5-minute / 16GB / CPU-only /
no-network constraint at 100K candidates -- TF-IDF+SVD fit/transform over
100K short documents and the subsequent per-candidate feature scoring both
run in well under a minute on a single CPU core in testing on this much more
constrained sandbox (1 vCPU, 4GB RAM).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.jd_parser import load_jd_from_file, JDRequirements
from src.semantic_matcher import SemanticMatcher
from src.scorer import score_candidate
from src.reasoning_generator import generate_reasoning
from src import feature_extractor as fx


def load_candidates(path: str) -> list[dict]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the Redrob JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument(
        "--jd", default=None,
        help="Path to job_description.md/.txt (default: docs/job_description.md next to this script)",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=100, help="Number of ranked candidates to output")
    args = parser.parse_args()

    t0 = time.time()

    repo_root = Path(__file__).resolve().parent
    jd_path = args.jd or str(repo_root / "docs" / "job_description.md")

    print(f"[rank.py] Loading JD from {jd_path}")
    jd: JDRequirements = load_jd_from_file(jd_path)

    print(f"[rank.py] Loading candidates from {args.candidates}")
    candidates = load_candidates(args.candidates)
    print(f"[rank.py] Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    # --- Build the joint semantic space ------------------------------------
    t1 = time.time()
    documents = [fx.candidate_full_text(c) for c in candidates]
    matcher = SemanticMatcher(n_components=120)
    candidate_vectors = matcher.fit_transform(documents)
    query_vector = matcher.transform([jd.semantic_text])[0]
    similarities = SemanticMatcher.similarity_to_query(candidate_vectors, query_vector)
    print(f"[rank.py] Semantic space fit + transform in {time.time()-t1:.1f}s")

    # --- Score every candidate -----------------------------------------------
    t2 = time.time()
    scored = []
    for cand, sim in zip(candidates, similarities):
        sc = score_candidate(cand, jd, semantic_similarity=float(sim))
        scored.append((sc, cand))
    print(f"[rank.py] Scored {len(scored)} candidates in {time.time()-t2:.1f}s")

    # --- Sort: rounded score desc, then candidate_id asc as deterministic
    # tiebreak. We sort on the *rounded* score (4 decimals) because that's
    # the value actually written to the CSV and checked by the validator --
    # sorting on the unrounded float can put two candidates that round to
    # the same displayed score in the wrong candidate_id order.
    scored.sort(key=lambda pair: (-round(pair[0].score, 4), pair[0].candidate_id))

    top = scored[: args.top_k]

    # --- Generate reasoning + write CSV ---------------------------------------
    n = len(top)
    best_score = top[0][0].score if top else 1.0
    rows = []
    for idx, (sc, cand) in enumerate(top):
        rank = idx + 1
        # Band by how close this candidate's score is to the best score in
        # the top-100, not just by rank position -- in a narrow, high-quality
        # pool (as this JD's "ideal candidate" note predicts) scores cluster
        # tightly, and a rank-90 candidate can still be a genuinely strong
        # match relative to the JD, just edged out by slightly better fits.
        relative = sc.score / best_score if best_score > 0 else 0.0
        band = "top" if relative >= 0.85 else ("mid" if relative >= 0.70 else "low")

        must_have_scores = {
            k: v for k, v in zip(
                jd.must_have_groups.keys(),
                [sc.components.get("must_have_skill_avg", 0.0)] * len(jd.must_have_groups),
            )
        }
        # Recompute per-group match (scorer only stored the averaged value)
        must_have_scores = fx.skill_group_match(cand, jd.must_have_groups)

        reasoning = generate_reasoning(
            cand, must_have_scores, sc.flags, sc.score, band
        )

        rows.append({
            "candidate_id": sc.candidate_id,
            "rank": rank,
            "score": round(sc.score, 4),
            "reasoning": reasoning,
        })

    # Enforce strictly non-increasing score by rank in the output (guards
    # against float rounding flips at tie boundaries).
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    n_honeypots_in_top = sum(1 for sc, _ in top if sc.is_honeypot)
    print(f"[rank.py] Wrote {len(rows)} rows to {out_path}")
    print(f"[rank.py] Honeypots in top {args.top_k}: {n_honeypots_in_top}")
    print(f"[rank.py] Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
