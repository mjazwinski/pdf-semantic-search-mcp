"""Local sparse embedding model using fastembed + SPLADE++.

SPLADE (Sparse Lexical And Expansion) produces sparse vectors over the
vocabulary of a transformer model.  Each non-zero dimension corresponds to a
vocabulary token; the value encodes that token's relevance to the input text.

This enables *term-level* retrieval (similar to BM25) that complements
the dense *semantic* vectors from :mod:`sentence_transformer`.  Combining
both via Qdrant's Reciprocal Rank Fusion (RRF) gives best-of-both-worlds
results: queries like "hot engine" and "how to tell that engine is hot" both
retrieve documents that discuss engine temperature.

All inference runs locally — the ONNX model is downloaded once (≈ 500 MB for
``Qdrant/SPLADE_PP_en_v1``) and cached in ``~/.cache/huggingface/``.  No API
key or network access is needed after the initial download.

Typical usage
-------------
    from pdf_semantic_search.embeddings.sparse_embedder import FastEmbedSparseEmbedder

    embedder = FastEmbedSparseEmbedder()
    sparse_vecs = embedder.embed_sparse(["hot engine", "coolant temperature sensor"])
    # → list of two SparseEmbedding objects with non-zero (index, value) pairs
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pdf_semantic_search.models import SparseEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface (Protocol)
# ---------------------------------------------------------------------------


@runtime_checkable
class SparseEmbeddingModel(Protocol):
    """Structural interface every sparse embedding back-end must satisfy.

    Any class that implements ``embed_sparse(texts) -> list[SparseEmbedding]``
    is automatically compatible — no explicit inheritance needed.
    """

    def embed_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        """Return one sparse embedding per input text.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            A list of :class:`~pdf_semantic_search.models.SparseEmbedding`
            objects in the same order as *texts*.  Each object holds the
            non-zero ``(indices, values)`` pairs of the high-dimensional
            sparse vector.
        """
        ...


# ---------------------------------------------------------------------------
# Default implementation — fastembed SPLADE++
# ---------------------------------------------------------------------------


class FastEmbedSparseEmbedder:
    """Encode text as sparse SPLADE vectors using Qdrant's ``fastembed`` library.

    The underlying ONNX model is *lazy-loaded* on the first call to
    :meth:`embed_sparse`.  Subsequent calls reuse the loaded model.

    Args:
        model_name: A ``fastembed``-compatible sparse model identifier.
                    Defaults to ``"prithivida/Splade_PP_en_v1"`` — a compact,
                    high-quality SPLADE++ variant that runs locally via ONNX.
        batch_size: Number of texts processed per ONNX inference pass.
    """

    def __init__(
        self,
        model_name: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # loaded lazily on first embed_sparse() call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Instantiate the fastembed SparseTextEmbedding model (called once)."""
        from fastembed import SparseTextEmbedding  # heavy import — lazy

        logger.info("Loading sparse embedding model %r …", self.model_name)
        self._model = SparseTextEmbedding(
            model_name=self.model_name,
            batch_size=self.batch_size,
        )
        logger.info("Sparse model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode *texts* and return a sparse SPLADE vector per string.

        Args:
            texts: One or more strings to encode.

        Returns:
            A list of :class:`~pdf_semantic_search.models.SparseEmbedding`
            objects (``indices`` + ``values`` as plain Python lists).

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must contain at least one string.")

        if self._model is None:
            self._load()

        logger.debug("Sparse-embedding %d text(s).", len(texts))

        # fastembed returns an iterator of SparseEmbedding namedtuples
        # (each has .indices: np.ndarray[int32] and .values: np.ndarray[float32])
        raw = list(self._model.embed(texts))  # type: ignore[union-attr]

        return [
            SparseEmbedding(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in raw
        ]
