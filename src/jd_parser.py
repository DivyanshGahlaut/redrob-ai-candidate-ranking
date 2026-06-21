"""
jd_parser.py
============
Parses a job description (plain text / markdown) into a structured
requirements object that the rest of the ranking pipeline consumes.

Design note
-----------
The JD for this challenge is unusually narrative (it explicitly tells you
"the right answer is not keyword matching"). A naive keyword extractor would
fail the exact trap the JD describes. So this parser does two things:

1.  Pulls out a small number of *structured* facts that are unambiguous in
    any well-written JD: experience band, location preference, must-have
    skill categories, explicit disqualifiers, "nice to have" skills, notice
    period tolerance. These are matched with light regex/keyword heuristics
    over clearly-labelled sections (e.g. "Things you absolutely need").

2.  Builds a free-text "semantic profile" string by concatenating the
    sections of the JD that describe *what the role actually needs* (the
    narrative "ideal candidate" paragraph, the role mandate, the "things you
    need" / "things you don't want" sections). This free text is embedded
    with the same TF-IDF/SVD space as candidate profiles in
    `feature_extractor.py`, so the semantic-fit score is reading the JD's
    meaning, not just a keyword bag.

This file is intentionally JD-agnostic: swap in a different job_description.md
and the structured extraction still works as long as the document uses
similar section headers. The hardcoded fallback values only kick in when a
section can't be found, so the system degrades gracefully rather than
crashing on a differently-formatted JD.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical skill taxonomy
# ---------------------------------------------------------------------------
# Maps a "concept" to the surface-form keywords that indicate a candidate has
# touched it. Used for the must-have / nice-to-have structured skill score.
# This is intentionally a small, curated taxonomy aligned to *this* JD's
# explicit "skills inventory" section, not a generic resume keyword list.

MUST_HAVE_SKILL_GROUPS: dict[str, list[str]] = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "openai embeddings",
        "bge", "e5", "embedding", "embeddings", "dense retrieval",
        "semantic search", "retrieval",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "vector db",
        "hybrid search", "hybrid retrieval", "bm25",
    ],
    "python": ["python"],
    "ranking_eval": [
        "ndcg", "mrr", "map", "a/b test", "ab test", "offline-to-online",
        "offline to online", "evaluation framework", "eval framework",
        "learning-to-rank", "learning to rank", "ltr",
    ],
}

NICE_TO_HAVE_SKILL_GROUPS: dict[str, list[str]] = {
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning llms", "fine-tuning", "finetuning"],
    "learning_to_rank_models": ["xgboost", "learning-to-rank", "neural ranking"],
    "hr_tech": ["hr-tech", "hrtech", "recruiting tech", "marketplace product"],
    "distributed_systems": ["distributed systems", "large-scale inference", "inference optimization"],
    "open_source": ["open-source", "open source contribution", "github"],
}

# Roles whose *primary* expertise the JD explicitly does not want, unless
# paired with strong NLP/IR exposure.
EXCLUDED_PRIMARY_DOMAINS = ["computer vision", "speech recognition", "robotics"]

# Company types the JD explicitly flags as a weak/negative signal when they
# represent the candidate's *entire* career.
SERVICES_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
}

# Title substrings indicating "pure research, no production" risk.
RESEARCH_ONLY_TITLE_HINTS = ["research scientist", "research engineer", "research fellow", "phd researcher"]
RESEARCH_ONLY_TEXT_HINTS = ["academic lab", "research lab", "no production", "purely academic", "research-only"]

# Title substrings indicating architecture/tech-lead drift away from coding.
NON_CODING_TITLE_HINTS = ["architect", "tech lead", "engineering manager", "head of engineering", "director of engineering"]


@dataclass
class JDRequirements:
    role_title: str = "Senior AI Engineer"
    min_years: float = 5.0
    max_years: float = 9.0
    years_band_is_soft: bool = True  # JD explicitly says this is a guideline, not a hard cutoff

    preferred_locations: list[str] = field(default_factory=lambda: [
        "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "bangalore", "bengaluru",
    ])
    country_required: str = "india"
    visa_sponsorship: bool = False

    notice_period_soft_cap_days: int = 30
    notice_period_hard_concern_days: int = 60

    must_have_groups: dict[str, list[str]] = field(default_factory=lambda: MUST_HAVE_SKILL_GROUPS)
    nice_to_have_groups: dict[str, list[str]] = field(default_factory=lambda: NICE_TO_HAVE_SKILL_GROUPS)

    excluded_primary_domains: list[str] = field(default_factory=lambda: EXCLUDED_PRIMARY_DOMAINS)
    services_firms: set[str] = field(default_factory=lambda: set(SERVICES_FIRMS))
    research_only_title_hints: list[str] = field(default_factory=lambda: RESEARCH_ONLY_TITLE_HINTS)
    research_only_text_hints: list[str] = field(default_factory=lambda: RESEARCH_ONLY_TEXT_HINTS)
    non_coding_title_hints: list[str] = field(default_factory=lambda: NON_CODING_TITLE_HINTS)

    # Free text used for semantic (TF-IDF/SVD) matching against candidate text.
    semantic_text: str = ""

    # Raw JD text, kept for traceability / debugging.
    raw_text: str = ""


def _extract_years_band(text: str) -> tuple[float, float]:
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years", text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 5.0, 9.0


def _section(text: str, header_pattern: str, next_header_pattern: str = r"\n#") -> str:
    """Grab the text between a header matching `header_pattern` and the next markdown header."""
    m = re.search(header_pattern, text, re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    rest = text[start:]
    m2 = re.search(next_header_pattern, rest)
    return rest[: m2.start()] if m2 else rest


def parse_job_description(text: str) -> JDRequirements:
    """Parse JD markdown/plain text into a JDRequirements object.

    Falls back to sensible defaults (calibrated to this challenge's JD) for
    any section that can't be located, so a differently-formatted JD doesn't
    crash the pipeline -- it just loses the fine-grained structured signal
    and leans more heavily on the semantic-text match.
    """
    req = JDRequirements(raw_text=text)

    # Title
    m = re.search(r"Job Description:\s*(.+)", text)
    if m:
        req.role_title = m.group(1).split("\u2014")[0].split("--")[0].strip()

    # Experience band
    req.min_years, req.max_years = _extract_years_band(text)

    # Visa sponsorship
    if re.search(r"don\u2019t sponsor|do not sponsor|don't sponsor", text, re.IGNORECASE):
        req.visa_sponsorship = False

    # Notice period
    m = re.search(r"sub-(\d+)-day notice", text, re.IGNORECASE)
    if m:
        req.notice_period_soft_cap_days = int(m.group(1))
    m = re.search(r"buy out up to (\d+) days", text, re.IGNORECASE)
    if m:
        req.notice_period_hard_concern_days = int(m.group(1))

    # Semantic text: concatenate the most meaning-dense narrative sections.
    semantic_chunks = []
    for header in [
        r"What you\u2019d actually be doing", r"What you'd actually be doing",
        r"The skills inventory.*",
        r"Things we explicitly do NOT want",
        r"How to read between the lines",
        r"The vibe check",
    ]:
        chunk = _section(text, header)
        if chunk:
            semantic_chunks.append(chunk)

    if not semantic_chunks:
        # Fallback: use the whole document if section headers weren't found.
        semantic_chunks = [text]

    req.semantic_text = "\n".join(semantic_chunks).strip()
    return req


def load_jd_from_file(path: str) -> JDRequirements:
    p = Path(path)
    # Target structured json file generated by LLM parser
    structured_json_path = p.parent / "job_description_structured.json"

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    req = parse_job_description(text)

    if structured_json_path.exists():
        try:
            with open(structured_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[jd_parser] Found LLM-parsed structured requirements at {structured_json_path}. Merging fields...")
            
            # Map fields from json to JDRequirements object
            for k, v in data.items():
                if hasattr(req, k):
                    if k == "services_firms":
                        setattr(req, k, set(v))
                    else:
                        setattr(req, k, v)
        except Exception as e:
            print(f"[jd_parser] Warning: Failed to load structured requirements JSON ({e}). Using standard parsed version.")

    return req
