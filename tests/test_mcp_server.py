"""Tests for the MCP server tool handlers.

All tests patch ``_get_deps()`` so no real embedding model or Qdrant instance
is required.  The module-level singleton is reset between tests via
``_reset_deps()``.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdf_semantic_search.mcp_server.server import (
    _get_deps,
    _handle_list_categories,
    _handle_search_docs,
    _reset_deps,
    _result_to_dict,
    _Deps,
)
from pdf_semantic_search.vector_store.qdrant_store import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Ensure the module-level _deps singleton is clean for every test."""
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


@pytest.mark.asyncio
async def test_search_docs_returns_single_text_content():
    deps = _make_deps(search_results=[_make_result()])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_search_docs({"query": {"text": "attention mechanism"}})

    assert len(results) == 1
    assert results[0].type == "text"


@pytest.mark.asyncio
async def test_search_docs_response_is_valid_json():
    deps = _make_deps(search_results=[_make_result()])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_search_docs({"query": {"text": "test query"}})

    parsed = json.loads(results[0].text)
    assert isinstance(parsed, list)


@pytest.mark.asyncio
async def test_search_docs_result_structure():
    deps = _make_deps(search_results=[_make_result(score=0.88, text="chunk", page=2)])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_search_docs({"query": {"text": "q"}})

    item = json.loads(results[0].text)[0]
    assert item["score"] == 0.88
    assert item["text"] == "chunk"
    assert item["page"] == 2


@pytest.mark.asyncio
async def test_search_docs_empty_results_returns_empty_array():
    deps = _make_deps(search_results=[])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_search_docs({"query": {"text": "nothing matches"}})

    assert json.loads(results[0].text) == []


@pytest.mark.asyncio
async def test_search_docs_embeds_query_text():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "my search query"}})

    deps.embedder.embed.assert_called_once_with(["my search query"])


@pytest.mark.asyncio
async def test_search_docs_passes_vector_to_store():
    fake_vector = [0.9, 0.8, 0.7, 0.6]
    deps = _make_deps(embed_vector=fake_vector)
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "q"}})

    deps.store.search.assert_called_once()
    call_kwargs = deps.store.search.call_args.kwargs
    assert call_kwargs["query_vector"] == fake_vector


@pytest.mark.asyncio
async def test_search_docs_passes_top_k():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "q"}, "max_results": 12})

    call_kwargs = deps.store.search.call_args.kwargs
    assert call_kwargs["top_k"] == 12


@pytest.mark.asyncio
async def test_search_docs_default_top_k_is_5():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "q"}})

    call_kwargs = deps.store.search.call_args.kwargs
    assert call_kwargs["top_k"] == 5


@pytest.mark.asyncio
async def test_search_docs_forwards_context_filter():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "q", "context": "report.pdf"}})

    assert deps.store.search.call_args.kwargs["context"] == "report.pdf"


@pytest.mark.asyncio
async def test_search_docs_forwards_category_filter():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "q", "category": "methods"}})

    assert deps.store.search.call_args.kwargs["category"] == "methods"


@pytest.mark.asyncio
async def test_search_docs_no_filter_passes_none():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_search_docs({"query": {"text": "q"}})

    kwargs = deps.store.search.call_args.kwargs
    assert kwargs["context"] is None
    assert kwargs["category"] is None


@pytest.mark.asyncio
async def test_search_docs_invalid_query_raises():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        with pytest.raises(Exception):  # Pydantic ValidationError
            await _handle_search_docs({"query": {}})  # missing required 'text'


@pytest.mark.asyncio
async def test_search_docs_multiple_results_ordered():
    results = [
        _make_result(score=0.95, text="best match"),
        _make_result(score=0.72, text="second match"),
        _make_result(score=0.51, text="third match"),
    ]
    deps = _make_deps(search_results=results)
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        response = await _handle_search_docs({"query": {"text": "q"}})

    parsed = json.loads(response[0].text)
    assert len(parsed) == 3
    assert parsed[0]["text"] == "best match"
    assert parsed[1]["text"] == "second match"


# ---------------------------------------------------------------------------
# _handle_list_categories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_categories_returns_single_text_content():
    deps = _make_deps(categories=["intro", "methods"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_list_categories()

    assert len(results) == 1
    assert results[0].type == "text"


@pytest.mark.asyncio
async def test_list_categories_response_is_valid_json():
    deps = _make_deps(categories=["intro", "methods"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_list_categories()

    parsed = json.loads(results[0].text)
    assert isinstance(parsed, list)


@pytest.mark.asyncio
async def test_list_categories_returns_all_categories():
    cats = ["appendix", "conclusion", "intro", "methods"]
    deps = _make_deps(categories=cats)
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_list_categories()

    assert json.loads(results[0].text) == cats


@pytest.mark.asyncio
async def test_list_categories_empty_collection():
    deps = _make_deps(categories=[])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        results = await _handle_list_categories()

    assert json.loads(results[0].text) == []


@pytest.mark.asyncio
async def test_list_categories_calls_store_method():
    deps = _make_deps(categories=["a"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        await _handle_list_categories()

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

    srv._deps = _make_deps()  # inject a fake
    _reset_deps()
    assert srv._deps is None


def test_get_deps_initialises_on_first_call():
    """_get_deps() must build embedder + store when _deps is None."""
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
# call_tool dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_routes_search_docs():
    deps = _make_deps()
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        from pdf_semantic_search.mcp_server.server import call_tool
        results = await call_tool("search_docs", {"query": {"text": "hello"}})
    assert len(results) == 1


@pytest.mark.asyncio
async def test_call_tool_routes_list_categories():
    deps = _make_deps(categories=["intro"])
    with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
        from pdf_semantic_search.mcp_server.server import call_tool
        results = await call_tool("list_categories", {})
    assert json.loads(results[0].text) == ["intro"]


@pytest.mark.asyncio
async def test_call_tool_raises_on_unknown_tool():
    from pdf_semantic_search.mcp_server.server import call_tool
    with pytest.raises(ValueError, match="Unknown tool"):
        await call_tool("nonexistent_tool", {})
