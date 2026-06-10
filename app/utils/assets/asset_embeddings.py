"""Lazy-loaded sentence-transformer embeddings with numpy cache.

The model is loaded on first call (not at import time) so server startup
stays fast even if `sentence-transformers` is missing or the model has
not been downloaded yet.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = os.getenv("ASSET_DB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Module-level cache so we never load the model twice.
_model = None
_model_name: Optional[str] = None


def _get_model(model_name: Optional[str] = None):
    """Lazy-load the SentenceTransformer model."""
    global _model, _model_name
    name = model_name or _DEFAULT_MODEL_NAME
    if _model is not None and _model_name == name:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for AssetDatabase semantic search. "
            "Install with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading embedding model '%s' ...", name)
    _model = SentenceTransformer(name)
    _model_name = name
    logger.info("Embedding model '%s' ready (dim=%s).", name, _model.get_sentence_embedding_dimension())
    return _model


# ── Public API ───────────────────────────────────────────────────────


def embed_texts(
    texts: Sequence[str],
    model_name: Optional[str] = None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of texts into a 2-D float32 numpy array (N × dim).

    Parameters
    ----------
    texts : sequence of str
        The texts to embed.
    model_name : str, optional
        Override the default model name.
    batch_size : int
        Batch size for the encoder.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), embedding_dim)`` float32 matrix.
    """
    model = _get_model(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(
    text: str,
    model_name: Optional[str] = None,
) -> np.ndarray:
    """Embed a single query string into a 1-D float32 vector.

    The returned vector is L2-normalized.
    """
    model = _get_model(model_name)
    vec = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


def cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between a query vector and a corpus matrix.

    Both inputs are assumed to be L2-normalized (as produced by
    ``embed_texts`` / ``embed_query``), so the dot product *is* the
    cosine similarity.

    Parameters
    ----------
    query_vec : np.ndarray
        Shape ``(dim,)`` — a single query embedding.
    corpus_matrix : np.ndarray
        Shape ``(N, dim)`` — the corpus embeddings.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — similarity scores in ``[-1, 1]``.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    return (corpus_matrix @ query_vec.T).flatten()


# ── Persistence helpers ──────────────────────────────────────────────


def save_embeddings(matrix: np.ndarray, path: str | Path) -> None:
    """Save an embedding matrix to a ``.npy`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), matrix)
    logger.info("Saved %s embeddings to %s", matrix.shape[0], path)


def load_embeddings(path: str | Path) -> Optional[np.ndarray]:
    """Load a cached embedding matrix, or return ``None`` if not found."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        matrix = np.load(str(path))
        logger.info("Loaded %s cached embeddings from %s", matrix.shape[0], path)
        return matrix
    except Exception as exc:
        logger.warning("Failed to load embedding cache %s: %s", path, exc)
        return None
