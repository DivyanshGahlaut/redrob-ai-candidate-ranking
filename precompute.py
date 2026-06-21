#!/usr/bin/env python3
"""
precompute.py
=============
Pre-computes candidate profile embeddings using a local Sentence Transformer model
(all-MiniLM-L6-v2) and saves them as a numpy binary file (.npy) at `data/candidate_embeddings.npy`.

Optimized for multi-core CPUs by explicitly configuring PyTorch threads.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Configure PyTorch thread settings before importing heavy libraries
import torch
cpu_cores = os.cpu_count() or 4
num_threads = cpu_cores
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(2)
print(f"[precompute.py] Configured PyTorch to use {num_threads} CPU threads.")

# Add workspace root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.feature_extractor import candidate_full_text

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is not installed. Please run `pip install -r requirements.txt` first.")
    sys.exit(1)


def load_candidates(path: str) -> list[dict]:
    candidates = []
    print(f"Loading candidates from {path}...")
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")
    return candidates


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings.")
    parser.add_argument(
        "--candidates",
        default="pub_dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl",
        help="Path to candidates.jsonl file"
    )
    parser.add_argument(
        "--out",
        default="data/candidate_embeddings.npy",
        help="Path to save the output numpy array"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence Transformer model name to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,  # Increased batch size to maximize CPU vectorization efficiency
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidates to process (useful for fast testing)"
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parent
    candidates_path = Path(args.candidates)
    if not candidates_path.is_absolute():
        candidates_path = repo_root / candidates_path

    if not candidates_path.exists():
        search_paths = list(repo_root.glob("**/candidates.jsonl"))
        if search_paths:
            candidates_path = search_paths[0]
            print(f"Candidates file found at: {candidates_path}")
        else:
            print(f"Error: candidates file not found at {candidates_path}")
            sys.exit(1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(str(candidates_path))
    if args.limit:
        print(f"Limiting candidate processing to the first {args.limit} candidates.")
        candidates = candidates[:args.limit]

    print(f"Loading SentenceTransformer model '{args.model}'...")
    t0 = time.time()
    model = SentenceTransformer(args.model)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print("Extracting full text profiles for candidates...")
    t0 = time.time()
    documents = [candidate_full_text(c) for c in candidates]
    print(f"Extracted {len(documents)} profiles in {time.time()-t0:.1f}s")

    print(f"Encoding profiles (batch_size={args.batch_size}) to embeddings...")
    t0 = time.time()
    
    # We run the standard encode with optimized thread count.
    # Note: multi_process encoding on CPU can sometimes lead to deadlock on Windows depending on python setup,
    # so optimized single-process multi-threaded PyTorch is much safer and highly efficient.
    embeddings = model.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    duration = time.time() - t0
    print(f"Encoded {len(embeddings)} profiles in {duration:.1f}s ({len(embeddings)/duration:.1f} profiles/sec)")

    print(f"Saving embeddings array of shape {embeddings.shape} to {out_path}...")
    np.save(str(out_path), embeddings)
    print("Pre-computation completed successfully!")


if __name__ == "__main__":
    main()
