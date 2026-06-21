import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.semantic_matcher import SemanticMatcher


def test_fallback_mode():
    """Ensure that if candidate_embeddings.npy is not loaded, it falls back to TF-IDF."""
    print("Testing Fallback Mode (TF-IDF + SVD)...")
    # Force fallback by specifying a non-existent path
    matcher = SemanticMatcher(embeddings_path="non_existent_file.npy")
    assert not matcher.use_transformer
    assert matcher.vectorizer is not None
    assert matcher.svd is not None

    documents = [
        "AI engineer experienced in embeddings and vector databases like pinecone.",
        "Marketing manager focused on social media campaigns and growth hacking.",
        "Backend engineer building microservices with Python and Postgres.",
        "Data scientist with machine learning models and pandas analysis.",
        "DevOps specialist automating cloud infrastructure on AWS and Kubernetes.",
        "Frontend developer using React and CSS for premium user experiences.",
        "HR director coordinating recruitment operations and talent acquisition.",
        "AI researcher exploring deep learning, NLP, and reinforcement learning.",
        "Product manager defining roadmap for SaaS enterprise platforms.",
        "Mobile developer building iOS and Android apps using Flutter."
    ]
    matcher.fit(documents)
    vectors = matcher.transform(documents)
    assert vectors.shape[0] == 10

    query = matcher.transform(["vector database search"])
    assert query.shape[0] == 1

    sims = SemanticMatcher.similarity_to_query(vectors, query[0])
    assert len(sims) == 10
    assert all(0.0 <= s <= 1.0 for s in sims)
    print("Fallback Mode Test: PASS")


def test_transformer_faiss_mode():
    """Ensure that if candidate_embeddings.npy exists, it initializes Transformer + FAISS search."""
    print("Testing Transformer/FAISS Mode...")
    embeddings_file = Path(__file__).resolve().parent.parent / "data" / "candidate_embeddings.npy"
    if not embeddings_file.exists():
        print("Skipping Transformer/FAISS Mode test (embeddings file not precomputed).")
        return

    matcher = SemanticMatcher()
    assert matcher.use_transformer
    assert matcher.model is not None
    assert matcher.faiss_index is not None
    assert matcher.embeddings is not None

    documents = ["dummy resume text"] * 5  # fit_transform returns precomputed embeddings
    vectors = matcher.fit_transform(documents)
    assert vectors.shape[1] == 384  # all-MiniLM-L6-v2 dimensionality

    query_text = "Experienced in vector database Pinecone and NDCG evaluation"
    query = matcher.transform([query_text])[0]
    assert query.shape == (384,)

    sims = SemanticMatcher.similarity_to_query(vectors, query)
    assert len(sims) == len(vectors)
    assert all(0.0 <= s <= 1.0 for s in sims)
    print("Transformer/FAISS Mode Test: PASS")


if __name__ == "__main__":
    test_fallback_mode()
    test_transformer_faiss_mode()
    print("\nAll SemanticMatcher tests passed.")
