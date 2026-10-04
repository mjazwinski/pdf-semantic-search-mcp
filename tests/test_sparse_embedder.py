"""Tests for FastEmbedSparseEmbedder and the SparseEmbeddingModel protocol.

Unit tests inject a fake fastembed model so no ONNX model download is needed.
The integration test (marked ``@pytest.mark.integration``) exercises the real
``Qdrant/SPLADE_PP_en_v1`` model and requires an internet connection on first
run (subsequent runs use the local cache).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_semantic_search.embeddings.sparse_embedder import (
    FastEmbedSparseEmbedder,
    SparseEmbeddingModel,
)
from pdf_semantic_search.models import SparseEmbedding


# ---------------------------------------------------------------------------
# Helpers — fake fastembed model
# ---------------------------------------------------------------------------


def _fake_fastembed_result(indices: list[int], values: list[float]) -> MagicMock:
    """Mimic a fastembed SparseEmbedding namedtuple (has .indices / .values)."""
    r = MagicMock()
    r.indices = np.array(indices, dtype=np.int32)
    r.values = np.array(values, dtype=np.float32)
    return r


def _make_embedder_with_fake_model(
    texts: list[str],
    indices: list[int] | None = None,
    values: list[float] | None = None,
) -> FastEmbedSparseEmbedder:
    """Return an embedder whose internal fastembed model is a MagicMock."""
    embedder = FastEmbedSparseEmbedder(model_name="test/model")
    fake_model = MagicMock()
    # embed() returns one fake result per text
    fake_result = _fake_fastembed_result(
        indices or [1, 10, 100],
        values or [0.9, 0.5, 0.3],
    )
    fake_model.embed.return_value = [fake_result] * len(texts)
    embedder._model = fake_model
    return embedder


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------


def test_embedder_satisfies_protocol():
    embedder = FastEmbedSparseEmbedder()
    assert isinstance(embedder, SparseEmbeddingModel)


# ---------------------------------------------------------------------------
# embed_sparse — output type and shape
# ---------------------------------------------------------------------------


def test_embed_sparse_returns_list_of_sparse_embeddings():
    embedder = _make_embedder_with_fake_model(["hello", "world"])
    results = embedder.embed_sparse(["hello", "world"])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, SparseEmbedding) for r in results)


def test_embed_sparse_single_text_returns_one_item():
    embedder = _make_embedder_with_fake_model(["single text"])
    results = embedder.embed_sparse(["single text"])
    assert len(results) == 1


def test_embed_sparse_indices_are_plain_python_list_of_ints():
    embedder = _make_embedder_with_fake_model(["test"])
    result = embedder.embed_sparse(["test"])[0]
    assert isinstance(result.indices, list)
    assert all(isinstance(i, int) for i in result.indices)


def test_embed_sparse_values_are_plain_python_list_of_floats():
    embedder = _make_embedder_with_fake_model(["test"])
    result = embedder.embed_sparse(["test"])[0]
    assert isinstance(result.values, list)
    assert all(isinstance(v, float) for v in result.values)


def test_embed_sparse_indices_and_values_have_same_length():
    embedder = _make_embedder_with_fake_model(["test"], indices=[1, 2, 3], values=[0.1, 0.2, 0.3])
    result = embedder.embed_sparse(["test"])[0]
    assert len(result.indices) == len(result.values)


def test_embed_sparse_values_match_fake_model_output():
    embedder = _make_embedder_with_fake_model(["test"], indices=[5, 42], values=[0.8, 0.4])
    result = embedder.embed_sparse(["test"])[0]
    assert result.indices == [5, 42]
    assert result.values == pytest.approx([0.8, 0.4], abs=1e-4)


def test_embed_sparse_raises_on_empty_list():
    embedder = FastEmbedSparseEmbedder()
    embedder._model = MagicMock()  # skip real load
    with pytest.raises(ValueError, match="at least one string"):
        embedder.embed_sparse([])


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def test_embed_sparse_triggers_model_load():
    embedder = FastEmbedSparseEmbedder(model_name="test/model")
    assert embedder._model is None

    fake_model = MagicMock()
    fake_model.embed.return_value = [_fake_fastembed_result([0], [1.0])]

    # _load() does a lazy `from fastembed import SparseTextEmbedding` — patch at source
    with patch("fastembed.SparseTextEmbedding", return_value=fake_model):
        embedder.embed_sparse(["warmup"])

    assert embedder._model is fake_model


def test_model_loaded_only_once():
    """embed_sparse() called twice must not reload the model."""
    embedder = _make_embedder_with_fake_model(["a"])
    embedder.embed_sparse(["a"])
    embedder.embed_sparse(["b"])
    # The _model.embed was called twice but _model itself was set once (inject)
    assert embedder._model.embed.call_count == 2


# ---------------------------------------------------------------------------
# Integration — real SPLADE model (requires model download on first run)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_sparse_embed_returns_non_empty_vectors():
    embedder = FastEmbedSparseEmbedder(model_name="prithivida/Splade_PP_en_v1")
    results = embedder.embed_sparse(["hot engine coolant temperature"])
    assert len(results) == 1
    r = results[0]
    assert len(r.indices) > 0
    assert len(r.values) == len(r.indices)
    assert all(v > 0 for v in r.values), "SPLADE values should be positive"


@pytest.mark.integration
def test_real_sparse_embed_different_texts_produce_different_vectors():
    embedder = FastEmbedSparseEmbedder(model_name="prithivida/Splade_PP_en_v1")
    r1 = embedder.embed_sparse(["engine temperature"])[0]
    r2 = embedder.embed_sparse(["machine learning transformer"])[0]
    # The non-zero index sets should differ (different vocabulary activated)
    assert set(r1.indices) != set(r2.indices)
