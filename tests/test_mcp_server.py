"""Tests for the MCP server tool handlers.

All tests patch ``_get_deps()`` so no real embedding model or Qdrant instance
is required.  Async handlers are invoked via ``asyncio.run()`` so the suite
requires only plain ``pytest`` — no async plugin needed.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from pdf_semantic_search.mcp_server.server import (
    _Deps,
    _get_deps,
    _handle_list_categories,
    _handle_search_docs,
    _reset_deps,
    _result_to_dict,
)
from pdf_semantic_search.vector_store.qdrant_store import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run a coroutine synchronously — no pytest-asyncio needed."""
    return asyncio.run(coro)


def _make_result(
    score: float = 0.91,
    text: str = "Sample chunk text.",
    source_file: str = "report.pdf",
    page: int = 0,
    chunk_index: int = 0,
    context: str | None = "report.pdf",
    category: str | None = "intro",
) -> SearchResult:
    return SearchResult(
        score=score,
        text=text,
        source_file=source_file,
        page=page,
        chunk_index=chunk_index,
        context=context,
        category=category,
        metadata={},
    )


def _make_deps(
    search_results: list[SearchResult] | None = None,
    categories: list[str] | None = None,
    embed_vector: list[float] | None = None,
) -> _Deps:
    embedder = MagicMock()
    embedder.dim = 4
    embedder.embed.return_value = [embed_vector or [0.1, 0.2, 0.3, 0.4]]

    store = MagicMock()
    store.search.return_value = search_results or []
    store.list_categories.return_value = categories or []

    return _Deps(embedder=embedder, store=store)


@pytest.fixture(autouse=True)
def reset_singleton():
    _reset_deps()
    yield
    _reset_deps()


# ---------------------------------------------------------------------------
# _result_to_dict
# ---------------------------------------------------------------------------


def test_result_to_dict_contains_all_fields():
    r = _make_result()
    d = _result_to_dict(r)
    assert set(d.keys()) == {"score", "text", "source_file", "page", "chunk_index", "context", "category"}


def test_result_to_dict_score_rounded():
    r = _make_result(score=0.912345678)
    assert _result_to_dict(r)["score"] == 0.9123


def test_result_to_dict_none_fields_preserved():
    r = _make_result(context=None, category=None)
    d = _result_to_dict(r)
    assert d["context"] is None
    assert d["category"] is None


# ---------------------------------------------------------------------------
# _handle_search_docs
# ---------------------------------------------------------------------------


def test_search_docs_returns_single_text_content():
    deps = _make_deps(search_results=[_make_result()])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_search_docs({"query": {"text": "attention mechanism"}}))
    assert len(results) == 1
    assert results[0].type == "text"


def test_search_docs_response_is_valid_json():
    deps = _make_deps(search_results=[_make_result()])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_search_docs({"query": {"text": "test query"}}))
    parsed = json.loads(results[0].text)
    assert isinstance(parsed, list)


def test_search_docs_result_structure():
    deps = _make_deps(search_results=[_make_result(score=0.88, text="chunk", page=2)])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_search_docs({"query": {"text": "q"}}))
    item = json.loads(results[0].text)[0]
    assert item["score"] == 0.88
    assert item["text"] == "chunk"
    assert item["page"] == 2


def test_search_docs_empty_results_returns_empty_array():
    deps = _make_deps(search_results=[])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_search_docs({"query": {"text": "nothing matches"}}))
    assert json.loads(results[0].text) == []


def test_search_docs_embeds_query_text():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "my search query"}}))
    deps.embedder.embed.assert_called_once_with(["my search query"])


def test_search_docs_passes_vector_to_store():
    fake_vector = [0.9, 0.8, 0.7, 0.6]
    deps = _make_deps(embed_vector=fake_vector)
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "q"}}))
    call_kwargs = deps.store.search.call_args.kwargs
    assert call_kwargs["query_vector"] == fake_vector


def test_search_docs_passes_top_k():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "q"}, "max_results": 12}))
    assert deps.store.search.call_args.kwargs["top_k"] == 12


def test_search_docs_default_top_k_is_5():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "q"}}))
    assert deps.store.search.call_args.kwargs["top_k"] == 5


def test_search_docs_forwards_context_filter():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "q", "context": "report.pdf"}}))
    assert deps.store.search.call_args.kwargs["context"] == "report.pdf"


def test_search_docs_forwards_category_filter():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "q", "category": "methods"}}))
    assert deps.store.search.call_args.kwargs["category"] == "methods"


def test_search_docs_no_filter_passes_none():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_search_docs({"query": {"text": "q"}}))
    kwargs = deps.store.search.call_args.kwargs
    assert kwargs["context"] is None
    assert kwargs["category"] is None


def test_search_docs_invalid_query_raises():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        with pytest.raises(Exception):  # Pydantic ValidationError — missing 'text'
            run(_handle_search_docs({"query": {}}))


def test_search_docs_multiple_results_ordered():
    results = [
        _make_result(score=0.95, text="best match"),
        _make_result(score=0.72, text="second match"),
        _make_result(score=0.51, text="third match"),
    ]
    deps = _make_deps(search_results=results)
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        response = run(_handle_search_docs({"query": {"text": "q"}}))
    parsed = json.loads(response[0].text)
    assert len(parsed) == 3
    assert parsed[0]["text"] == "best match"
    assert parsed[1]["text"] == "second match"


# ---------------------------------------------------------------------------
# _handle_list_categories
# ---------------------------------------------------------------------------


def test_list_categories_returns_single_text_content():
    deps = _make_deps(categories=["intro", "methods"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_list_categories())
    assert len(results) == 1
    assert results[0].type == "text"


def test_list_categories_response_is_valid_json():
    deps = _make_deps(categories=["intro", "methods"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_list_categories())
    assert isinstance(json.loads(results[0].text), list)


def test_list_categories_returns_all_categories():
    cats = ["appendix", "conclusion", "intro", "methods"]
    deps = _make_deps(categories=cats)
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_list_categories())
    assert json.loads(results[0].text) == cats


def test_list_categories_empty_collection():
    deps = _make_deps(categories=[])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = run(_handle_list_categories())
    assert json.loads(results[0].text) == []


def test_list_categories_calls_store_method():
    deps = _make_deps(categories=["a"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        run(_handle_list_categories())
    deps.store.list_categories.assert_called_once()


# ---------------------------------------------------------------------------
# _get_deps singleton behaviour
# ---------------------------------------------------------------------------


def test_get_deps_returns_same_instance():
    fake_deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._deps", fake_deps):
        d1 = _get_deps()
        d2 = _get_deps()
    assert d1 is d2


def test_reset_deps_clears_singleton():
    import pdf_semantic_search.mcp_server.server as srv
    srv._deps = _make_deps()
    _reset_deps()
    assert srv._deps is None


def test_get_deps_initialises_on_first_call():
    mock_embedder = MagicMock()
    mock_embedder.dim = 4
    mock_store = MagicMock()

    with (
        patch("pdf_semantic_search.mcp_server.server.SentenceTransformerEmbedder", return_value=mock_embedder),
        patch("pdf_semantic_search.mcp_server.server.QdrantStore", return_value=mock_store),
    ):
        deps = _get_deps()

    assert deps.embedder is mock_embedder
    assert deps.store is mock_store
    mock_store._connect.assert_called_once()


# ---------------------------------------------------------------------------
# FastMCP tool wrappers — verify they delegate to the handler functions
# ---------------------------------------------------------------------------


def test_fastmcp_search_docs_delegates_to_handler():
    deps = _make_deps(search_results=[_make_result(score=0.88, text="NLP chunk")])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        from pdf_semantic_search.mcp_server.server import search_docs
        raw = run(search_docs(text="attention mechanism"))
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["text"] == "NLP chunk"
    assert parsed[0]["score"] == 0.88


def test_fastmcp_search_docs_forwards_filters():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        from pdf_semantic_search.mcp_server.server import search_docs
        run(search_docs(text="query", context="doc.pdf", category="intro", max_results=7))
    kwargs = deps.store.search.call_args.kwargs
    assert kwargs["context"] == "doc.pdf"
    assert kwargs["category"] == "intro"
    assert kwargs["top_k"] == 7


def test_fastmcp_list_categories_delegates_to_handler():
    deps = _make_deps(categories=["intro", "methods"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        from pdf_semantic_search.mcp_server.server import list_categories
        raw = run(list_categories())
    assert json.loads(raw) == ["intro", "methods"]


def test_main_disables_model_loading_progress_bars(tmp_path):
    import pdf_semantic_search.mcp_server.server as srv

    with (
        patch("pdf_semantic_search.mcp_server.server._disable_model_progress_bars") as disable,
        patch.object(srv.mcp, "run"),
        patch.object(srv.settings, "log_path", str(tmp_path / "server.log")),
    ):
        srv.main()

    disable.assert_called_once_with()
