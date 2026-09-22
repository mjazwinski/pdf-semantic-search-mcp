"""Tests for QdrantStore.

Unit tests mock the qdrant_client entirely so no Docker instance is needed.
Integration tests (marked ``@pytest.mark.integration``) require a live Qdrant
container — start one with ``docker compose up -d`` before running them.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from pdf_semantic_search.vector_store.qdrant_store import QdrantStore, SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # tiny dimension for tests


def _make_store() -> QdrantStore:
    return QdrantStore(host="localhost", port=6333, collection="test_col", dim=DIM)


def _fake_vector() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


def _make_mock_client(collection_exists: bool = False) -> MagicMock:
    """Build a MagicMock that looks like a QdrantClient."""
    client = MagicMock()

    # get_collections() response
    if collection_exists:
        col = MagicMock()
        col.name = "test_col"
        client.get_collections.return_value.collections = [col]
    else:
        client.get_collections.return_value.collections = []

    # scroll() — return one page then stop
    client.scroll.return_value = ([], None)

    return client


# ---------------------------------------------------------------------------
# _connect
# ---------------------------------------------------------------------------


def test_connect_creates_collection_when_missing():
    store = _make_store()
    mock_client = _make_mock_client(collection_exists=False)

    with patch("pdf_semantic_search.vector_store.qdrant_store.QdrantClient", return_value=mock_client):
        store._connect()

    mock_client.create_collection.assert_called_once()
    args, kwargs = mock_client.create_collection.call_args
    assert kwargs.get("collection_name") == "test_col" or args[0] == "test_col"


def test_connect_skips_creation_when_collection_exists():
    store = _make_store()
    mock_client = _make_mock_client(collection_exists=True)

    with patch("pdf_semantic_search.vector_store.qdrant_store.QdrantClient", return_value=mock_client):
        store._connect()

    mock_client.create_collection.assert_not_called()


def test_connect_is_idempotent():
    """Calling _connect() twice must not open a second client."""
    store = _make_store()
    mock_client = _make_mock_client(collection_exists=True)

    with patch("pdf_semantic_search.vector_store.qdrant_store.QdrantClient", return_value=mock_client):
        store._connect()
        store._connect()  # second call — should be a no-op

    # QdrantClient constructor called exactly once
    assert mock_client.get_collections.call_count == 1


def test_payload_indexes_created_for_new_collection():
    store = _make_store()
    mock_client = _make_mock_client(collection_exists=False)

    with patch("pdf_semantic_search.vector_store.qdrant_store.QdrantClient", return_value=mock_client):
        store._connect()

    indexed_fields = {
        c.kwargs.get("field_name") or c.args[1]
        for c in mock_client.create_payload_index.call_args_list
    }
    assert {"context", "category", "source_file"}.issubset(indexed_fields)


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


def _make_connected_store() -> tuple[QdrantStore, MagicMock]:
    store = _make_store()
    mock_client = _make_mock_client(collection_exists=True)
    store._client = mock_client  # inject pre-connected client
    return store, mock_client


def test_upsert_calls_qdrant_upsert():
    store, mock_client = _make_connected_store()
    points = [{"vector": _fake_vector(), "payload": {"text": "hello", "source_file": "a.pdf"}}]
    store.upsert(points)
    mock_client.upsert.assert_called_once()


def test_upsert_passes_wait_true():
    store, mock_client = _make_connected_store()
    store.upsert([{"vector": _fake_vector(), "payload": {"text": "x", "source_file": "a.pdf"}}])
    _, kwargs = mock_client.upsert.call_args
    assert kwargs.get("wait") is True


def test_upsert_auto_generates_id_when_missing():
    store, mock_client = _make_connected_store()
    store.upsert([{"vector": _fake_vector(), "payload": {"text": "x", "source_file": "a.pdf"}}])
    _, kwargs = mock_client.upsert.call_args
    point = kwargs["points"][0]
    assert point.id is not None


def test_upsert_uses_provided_id():
    store, mock_client = _make_connected_store()
    custom_id = str(uuid.uuid4())
    store.upsert([{"id": custom_id, "vector": _fake_vector(), "payload": {"text": "x", "source_file": "a.pdf"}}])
    _, kwargs = mock_client.upsert.call_args
    assert kwargs["points"][0].id == custom_id


def test_upsert_raises_on_empty_list():
    store, _ = _make_connected_store()
    with pytest.raises(ValueError, match="at least one item"):
        store.upsert([])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _make_scored_point(score: float, payload: dict) -> MagicMock:
    sp = MagicMock()
    sp.score = score
    sp.payload = payload
    return sp


def _make_query_response(points: list) -> MagicMock:
    """Shape a mock like qdrant-client ≥ 1.9 QueryResponse (has .points)."""
    resp = MagicMock()
    resp.points = points
    return resp


def test_search_returns_search_results():
    store, mock_client = _make_connected_store()
    mock_client.query_points.return_value = _make_query_response([
        _make_scored_point(0.9, {"text": "chunk1", "source_file": "a.pdf", "page": 0, "chunk_index": 0}),
        _make_scored_point(0.7, {"text": "chunk2", "source_file": "a.pdf", "page": 1, "chunk_index": 0}),
    ])

    results = store.search(_fake_vector(), top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].score == pytest.approx(0.9)
    assert results[0].text == "chunk1"


def test_search_no_filter_passes_none_to_qdrant():
    store, mock_client = _make_connected_store()
    mock_client.query_points.return_value = _make_query_response([])

    store.search(_fake_vector(), top_k=5, context=None, category=None)

    _, kwargs = mock_client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_search_context_filter_applied():
    store, mock_client = _make_connected_store()
    mock_client.query_points.return_value = _make_query_response([])

    store.search(_fake_vector(), top_k=5, context="doc.pdf")

    _, kwargs = mock_client.query_points.call_args
    flt = kwargs["query_filter"]
    assert flt is not None
    assert any(c.key == "context" for c in flt.must)


def test_search_category_filter_applied():
    store, mock_client = _make_connected_store()
    mock_client.query_points.return_value = _make_query_response([])

    store.search(_fake_vector(), top_k=5, category="introduction")

    _, kwargs = mock_client.query_points.call_args
    flt = kwargs["query_filter"]
    assert any(c.key == "category" for c in flt.must)


def test_search_both_filters_combined():
    store, mock_client = _make_connected_store()
    mock_client.query_points.return_value = _make_query_response([])

    store.search(_fake_vector(), top_k=3, context="doc.pdf", category="ch1")

    _, kwargs = mock_client.query_points.call_args
    flt = kwargs["query_filter"]
    keys = {c.key for c in flt.must}
    assert keys == {"context", "category"}


# ---------------------------------------------------------------------------
# list_categories
# ---------------------------------------------------------------------------


def test_list_categories_deduplicates_and_sorts():
    store, mock_client = _make_connected_store()

    page1 = [
        MagicMock(payload={"category": "intro"}),
        MagicMock(payload={"category": "methods"}),
        MagicMock(payload={"category": "intro"}),   # duplicate
    ]
    # First scroll call returns page1 with offset; second terminates the loop
    mock_client.scroll.side_effect = [
        (page1, "cursor-abc"),
        ([], None),
    ]

    result = store.list_categories()

    assert result == ["intro", "methods"]  # sorted, deduplicated


def test_list_categories_excludes_none_and_empty():
    store, mock_client = _make_connected_store()
    page1 = [
        MagicMock(payload={"category": None}),
        MagicMock(payload={"category": ""}),
        MagicMock(payload={"category": "results"}),
    ]
    mock_client.scroll.side_effect = [(page1, None)]

    result = store.list_categories()

    assert result == ["results"]


# ---------------------------------------------------------------------------
# Integration test — requires live Qdrant (docker compose up -d)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_upsert_and_search():
    import random

    store = QdrantStore(
        host="localhost", port=6333, collection="pytest_integration", dim=DIM
    )
    store._connect()

    vector = [random.random() for _ in range(DIM)]
    store.upsert([{
        "vector": vector,
        "payload": {
            "text": "integration test chunk",
            "source_file": "test.pdf",
            "page": 0,
            "chunk_index": 0,
            "context": "test.pdf",
            "category": "testing",
        },
    }])

    results = store.search(vector, top_k=1)
    assert len(results) == 1
    assert results[0].text == "integration test chunk"
    assert results[0].score > 0.99  # same vector → near-perfect match

    # Cleanup
    store._client.delete_collection("pytest_integration")
