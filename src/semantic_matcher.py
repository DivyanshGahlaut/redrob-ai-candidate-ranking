"""
semantic_matcher.py
=====================
Lightweight, CPU-only, network-free semantic similarity between the JD's
meaning-dense text and each candidate's full free-text profile.

This module has been enhanced to support two modes:
1.  **Transformer Mode (Sentence Transformers + FAISS):**
    If pre-computed embeddings exist at `data/candidate_embeddings.npy`, this class
    loads them, initialises a FAISS index (`faiss.IndexFlatIP` over L2-normalised vectors)
    for high-speed retrieval, and encodes the query JD on-the-fly using `all-MiniLM-L6-v2`.
    This brings state-of-the-art dense semantic search to the pipeline while complying
    with the CPU-only, offline sandboxing constraints during ranking.
2.  **Fallback Mode (TF-IDF + TruncatedSVD / LSA):**
    If pre-computed embeddings are not present, it degrades gracefully to a TF-IDF
    vocabulary vector space mapped to 120 latent dimensions, maintaining backward
    compatibility and requiring no pre-computation steps.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np


class SemanticMatcher:
    def __init__(
        self,
        n_components: int = 120,
        max_features: int = 40000,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_path: str | None = None,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.model_name = model_name

        self.use_transformer = False
        self.embeddings = None
        self.model = None
        self.faiss_index = None

        # Resolve embeddings_path default
        repo_root = Path(__file__).resolve().parent.parent
        self.actual_embeddings_path = (
            Path(embeddings_path) if embeddings_path else repo_root / "data" / "candidate_embeddings.npy"
        )
        if not self.actual_embeddings_path.is_absolute():
            self.actual_embeddings_path = repo_root / self.actual_embeddings_path

        if self.actual_embeddings_path.exists():
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                print(f"[SemanticMatcher] Loading pre-computed embeddings from {self.actual_embeddings_path}...")
                self.embeddings = np.load(str(self.actual_embeddings_path))
                print(f"[SemanticMatcher] Loading SentenceTransformer model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name)

                # Normalize candidate embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)

                # Initialize FAISS IndexFlatIP (IP = Inner Product, which equals cosine similarity on normalized vectors)
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.faiss_index.add(self.embeddings)

                self.use_transformer = True
                print(f"[SemanticMatcher] Initialized FAISS index successfully with shape {self.embeddings.shape}.")
            except Exception as e:
                print(f"[SemanticMatcher] Warning: Failed to load precomputed embeddings or FAISS ({e}). Falling back to TF-IDF + SVD.")

        if not self.use_transformer:
            # Fallback to TF-IDF + TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            print("[SemanticMatcher] Initialising fallback TF-IDF + TruncatedSVD (LSA) space.")
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.6,
                sublinear_tf=True,
                stop_words="english",
            )
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            self._fitted = False

    def fit(self, documents: list[str]) -> "SemanticMatcher":
        if self.use_transformer:
            # No fitting required, embeddings are precomputed
            return self

        if len(documents) < 100:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0

        tfidf = self.vectorizer.fit_transform(documents)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self.use_transformer:
            # Encode query/documents using Sentence Transformer
            embeddings = self.model.encode(documents, convert_to_numpy=True)
            import faiss
            faiss.normalize_L2(embeddings)
            return embeddings

        if not self._fitted:
            raise RuntimeError("SemanticMatcher must be fit() before transform().")
        from sklearn.preprocessing import normalize

        tfidf = self.vectorizer.transform(documents)
        latent = self.svd.transform(tfidf)
        return normalize(latent)

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        if self.use_transformer:
            # Simply return the loaded, L2-normalized precomputed candidate embeddings
            return self.embeddings

        if len(documents) < 100:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0

        from sklearn.preprocessing import normalize

        tfidf = self.vectorizer.fit_transform(documents)
        latent = self.svd.fit_transform(tfidf)
        self._fitted = True
        return normalize(latent)

    @staticmethod
    def similarity_to_query(candidate_vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity (vectors are already L2-normalized), rescaled
        from [-1, 1] to [0, 1] for easy blending with other [0,1] scores.
        """
        sims = candidate_vectors @ query_vector.reshape(-1)
        return (sims + 1.0) / 2.0
