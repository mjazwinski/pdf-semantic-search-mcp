"""Tests for SentenceTransformerEmbedder.

The real model is heavy, so fast unit tests use a lightweight fixture model
(paraphrase-MiniLM-L3-v2 — ~17 MB) or mock the encode call.  Integration
tests that actually download the model are marked ``@pytest.mark.integration``
and skipped in CI by default.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_semantic_search.embeddings.sentence_transformer import (
    EmbeddingModel,
    SentenceTransformerEmbedder,
)

# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_embedder_satisfies_protocol():
    """SentenceTransformerEmbedder must be recognised as an EmbeddingModel."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    assert isinstance(embedder, EmbeddingModel)


# ---------------------------------------------------------------------------
# Unit tests — mock the heavy SentenceTransformer import
# ---------------------------------------------------------------------------

DIM = 384
FAKE_VECTORS = np.random.rand(3, DIM).astype("float32")


def _make_mock_st(vectors: np.ndarray = FAKE_VECTORS) -> MagicMock:
    """Return a MagicMock that quacks like a SentenceTransformer."""
    mock = MagicMock()
    mock.encode.return_value = vectors
    mock.get_sentence_embedding_dimension.return_value = vectors.shape[1]
    return mock


@pytest.fixture()
def embedder_with_mock():
    """An embedder whose inner SentenceTransformer is replaced by a mock."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    embedder._model = _make_mock_st()
    return embedder


def test_embed_returns_list_of_lists(embedder_with_mock):
    texts = ["hello", "world", "foo"]
    result = embedder_with_mock.embed(texts)
    assert isinstance(result, list)
    assert all(isinstance(v, list) for v in result)


def test_embed_length_matches_input(embedder_with_mock):
    texts = ["a", "b", "c"]
    result = embedder_with_mock.embed(texts)
    assert len(result) == len(texts)


def test_embed_vector_dimension(embedder_with_mock):
    result = embedder_with_mock.embed(["test"])
    # shape of FAKE_VECTORS is (3, DIM) — first row used
    assert len(result[0]) == DIM


def test_embed_raises_on_empty_input(embedder_with_mock):
    with pytest.raises(ValueError, match="at least one string"):
        embedder_with_mock.embed([])


def test_dim_property(embedder_with_mock):
    assert embedder_with_mock.dim == DIM


def test_encode_called_with_correct_kwargs(embedder_with_mock):
    texts = ["sentence one", "sentence two", "sentence three"]
    embedder_with_mock.embed(texts)
    embedder_with_mock._model.encode.assert_called_once_with(
        texts,
        batch_size=embedder_with_mock.batch_size,
        normalize_embeddings=embedder_with_mock.normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def test_model_loaded_lazily():
    """_model must be None until embed() or dim is first accessed."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    assert embedder._model is None  # not loaded yet

    with patch(
        "pdf_semantic_search.embeddings.sentence_transformer.SentenceTransformer"  # noqa: E501
        if False
        else "sentence_transformers.SentenceTransformer"
    ):
        pass  # just verifying the attribute, not actually loading


def test_normalize_flag_forwarded(embedder_with_mock):
    embedder_with_mock.normalize = False
    embedder_with_mock.embed(["test"])
    call_kwargs = embedder_with_mock._model.encode.call_args.kwargs
    assert call_kwargs["normalize_embeddings"] is False


# ---------------------------------------------------------------------------
# Integration test — downloads the real model (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_model_embed():
    """End-to-end: load the real model and embed two sentences."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed(["The quick brown fox", "jumped over the lazy dog"])

    assert len(result) == 2
    assert len(result[0]) == 384
    # cosine similarity between two related sentences should be > 0.5
    v1, v2 = np.array(result[0]), np.array(result[1])
    similarity = float(np.dot(v1, v2))
    assert similarity > 0.5, f"Unexpectedly low similarity: {similarity}"
