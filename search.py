"""
search.py
---------
Local semantic search using SentenceTransformers + FAISS.

Documents are encoded into dense vector embeddings.  Queries are encoded
with the same model and the nearest neighbours are retrieved via cosine
similarity (implemented as inner-product search on L2-normalised vectors).

No internet connection is required after the model is downloaded once.
The default model ('all-MiniLM-L6-v2') is ~90 MB and runs comfortably on CPU.

Performance notes
-----------------
- The SentenceTransformer model is loaded once via `load_embedding_model()`
  and cached at the module level so it survives across Streamlit reruns.
- The FAISS index is built per document set and cached separately in app.py
  using st.cache_resource keyed on document fingerprints (not full text).
"""

from __future__ import annotations

import hashlib
import numpy as np

# ---------------------------------------------------------------------------
# Module-level model cache — loaded once, reused forever
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, object] = {}


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Return a cached SentenceTransformer model.
    The model is loaded from disk/cache only on the very first call;
    subsequent calls return the already-loaded instance immediately.
    """
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        print(f"  Loading embedding model '{model_name}' …")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def fingerprint(texts: list[str]) -> str:
    """Return a short hash of the document texts — used as a cache key."""
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


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
        # Reuse the already-loaded model — no reload on every search
        self._model = load_embedding_model(model_name)
        self._build_index(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

        # Chunk long documents to avoid memory spikes; encode in one batch
        truncated = [t[:4000] for t in texts]  # ~1000 tokens max per doc
        embeddings: np.ndarray = self._model.encode(
            truncated, convert_to_numpy=True, show_progress_bar=False,
            batch_size=32,
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

    def best_passage(self, query: str, text: str, window: int = 300) -> str:
        """
        Find the most relevant passage within *text* for *query* by sliding
        a window over sentences and returning the highest-scoring chunk.

        Parameters
        ----------
        query : str
            The search query.
        text : str
            Full document text to search within.
        window : int
            Approximate character window size per passage chunk.

        Returns
        -------
        The best-matching passage string (up to *window* chars).
        """
        import re

        # Split into sentences, then group into overlapping chunks
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:window]

        # Build chunks of ~window chars
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= window:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)

        if not chunks:
            return text[:window]

        if len(chunks) == 1:
            return chunks[0]

        # Encode all chunks and find the best match
        chunk_embs: np.ndarray = self._model.encode(
            chunks, convert_to_numpy=True, show_progress_bar=False
        )
        import faiss as _faiss
        _faiss.normalize_L2(chunk_embs)

        q_emb: np.ndarray = self._model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )
        _faiss.normalize_L2(q_emb)

        scores = (chunk_embs @ q_emb.T).flatten()
        best_idx = int(scores.argmax())
        return chunks[best_idx]
