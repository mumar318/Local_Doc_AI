"""
search.py
---------
Local semantic search using SentenceTransformers + FAISS.

Documents are encoded into dense vector embeddings.  Queries are encoded
with the same model and the nearest neighbours are retrieved via cosine
similarity (implemented as inner-product search on L2-normalised vectors).

No internet connection is required after the model is downloaded once.
The default model ('all-MiniLM-L6-v2') is ~90 MB and runs comfortably on CPU.
"""

from __future__ import annotations

import numpy as np


class SemanticSearch:
    """
    Build an in-memory FAISS index from a list of document texts and
    support natural-language queries against it.

    Parameters
    ----------
    texts : list[str]
        The document texts to index (one entry per document).
    fnames : list[str]
        Corresponding filenames (same order as *texts*).
    model_name : str
        SentenceTransformers model identifier.  Defaults to the lightweight
        'all-MiniLM-L6-v2' which balances speed and quality well.
    """

    def __init__(
        self,
        texts: list[str],
        fnames: list[str],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if len(texts) != len(fnames):
            raise ValueError("texts and fnames must have the same length")

        self.fnames = fnames
        self._load_model(model_name)
        self._build_index(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str) -> None:
        """Load the SentenceTransformer model (downloads on first use)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        print(f"  Loading embedding model '{model_name}' …")
        self._model = SentenceTransformer(model_name)

    def _build_index(self, texts: list[str]) -> None:
        """Encode all documents and build a FAISS cosine-similarity index."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for semantic search. "
                "Install it with: pip install faiss-cpu"
            ) from exc

        print(f"  Encoding {len(texts)} document(s) …")
        embeddings: np.ndarray = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

        # L2-normalise so that inner-product == cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner Product on normalised vecs
        self._index.add(embeddings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Search for documents semantically similar to *query*.

        Parameters
        ----------
        query : str
            Natural-language search query.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list of (filename, similarity_score) tuples, sorted by descending
        similarity.  Scores are in the range [-1, 1]; higher is more similar.
        """
        import faiss

        top_k = min(top_k, len(self.fnames))
        q_emb: np.ndarray = self._model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(q_emb)

        similarities, indices = self._index.search(q_emb, top_k)

        results = [
            (self.fnames[idx], float(similarities[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx != -1  # FAISS returns -1 for empty slots
        ]
        return results
