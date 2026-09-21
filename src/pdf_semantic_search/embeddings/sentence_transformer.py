"""Local embedding model wrapper using SentenceTransformer.

All inference runs on the local machine — no API keys or network calls required
after the initial model download.

Typical usage
-------------
    from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    vectors = embedder.embed(["hello world", "semantic search rocks"])
    # → list of two float lists, each of length 384
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface (Protocol)
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingModel(Protocol):
    """Structural interface every embedding back-end must satisfy.

    Any class that implements ``embed(texts) -> list[list[float]]`` is
    automatically compatible — no explicit inheritance needed.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            A list of float vectors in the same order as *texts*.
        """
        ...


# ---------------------------------------------------------------------------
# Default implementation
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Encode text locally using a ``sentence-transformers`` model.

    The underlying :class:`sentence_transformers.SentenceTransformer` is
    *lazy-loaded* on the first call to :meth:`embed` so that importing this
    module never triggers a slow model download.

    Args:
        model_name: Any model name accepted by ``sentence-transformers``
                    (e.g. ``"all-MiniLM-L6-v2"``) or a local directory path.
        device:     PyTorch device string — ``"cpu"``, ``"cuda"``, ``"mps"``.
                    Pass ``None`` (default) to let the library auto-detect.
        batch_size: Number of sentences encoded in a single forward pass.
                    Larger values are faster on GPU; smaller values use less RAM.
        normalize:  If ``True`` (default) vectors are L2-normalised so that
                    cosine similarity equals dot product — required for Qdrant's
                    ``Cosine`` distance metric.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # loaded lazily on first embed() call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Instantiate the SentenceTransformer model (called once)."""
        from sentence_transformers import SentenceTransformer  # heavy import

        logger.info("Loading embedding model %r on device=%r …", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Model loaded — vector dim=%d", self.dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this model.

        Triggers model loading if the model has not been loaded yet.
        """
        if self._model is None:
            self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode *texts* and return plain Python float lists.

        Args:
            texts: One or more strings to encode.  Empty strings are
                   accepted but will produce low-quality embeddings.

        Returns:
            A list of ``len(texts)`` float lists, each of length :attr:`dim`.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must contain at least one string.")

        if self._model is None:
            self._load()

        logger.debug("Embedding %d text(s) with batch_size=%d", len(texts), self.batch_size)

        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # numpy ndarray → plain Python list[list[float]] for JSON-serialisability
        return vectors.tolist()
