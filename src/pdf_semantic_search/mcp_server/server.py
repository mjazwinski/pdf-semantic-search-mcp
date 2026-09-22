"""MCP server — exposes ``search_docs`` and ``list_categories`` over stdio.

Start with:
    pdf-mcp-server
    # or:
    python -m pdf_semantic_search.mcp_server.server

MCP client configuration (e.g. Claude Desktop / Cursor):
    {
      "mcpServers": {
        "pdf-search": {
          "command": "<absolute-path-to-pdf-mcp-server>"
        }
      }
    }

Exposed tools
-------------
search_docs(text, context?, category?, max_results?)
    Semantic search over ingested PDF chunks.
    ``context`` and ``category`` are optional Qdrant payload filters.
    Returns a JSON array of result objects:
        [{score, text, source_file, page, chunk_index, context, category}, ...]

list_categories()
    Returns a JSON array of all distinct ``category`` strings stored in Qdrant.
    Useful for letting an agent enumerate document sections before narrowing a search.

Dependency lifecycle
--------------------
Both the embedding model and the Qdrant client are expensive to initialise.
They are created once on the first tool call via :func:`_get_deps` and reused
for the lifetime of the server process.

API note
--------
This server uses ``MCPServer`` (mcp >= 2.0, formerly ``FastMCP`` in mcp 1.x).
The ``@mcp.tool()`` decorator infers the JSON schema directly from the
function signature and type annotations — no manual ``inputSchema`` needed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

from pdf_semantic_search.config import settings
from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder
from pdf_semantic_search.models import DocumentEntry, DocumentQuery
from pdf_semantic_search.vector_store.qdrant_store import QdrantStore, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised dependencies (created once, reused for server lifetime)
# ---------------------------------------------------------------------------


@dataclass
class _Deps:
    embedder: SentenceTransformerEmbedder
    store: QdrantStore


_deps: Optional[_Deps] = None  # module-level singleton


def _get_deps() -> _Deps:
    """Return the shared embedder + store, initialising them on first call."""
    global _deps
    if _deps is None:
        logger.info("Initialising MCP server dependencies …")

        embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            device=None,    # auto-detect CPU / CUDA / MPS
            normalize=True,
        )
        _ = embedder.dim  # trigger model load now; first search won't be slow

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            dim=embedder.dim,
        )
        store._connect()

        _deps = _Deps(embedder=embedder, store=store)
        logger.info(
            "Dependencies ready (model=%r, dim=%d).",
            settings.embedding_model,
            embedder.dim,
        )

    return _deps


def _reset_deps() -> None:
    """Replace the singleton with *None*.  Used in tests only."""
    global _deps
    _deps = None


# ---------------------------------------------------------------------------
# Tool handlers  (pure business logic — testable without MCPServer)
# ---------------------------------------------------------------------------


async def _handle_search_docs(arguments: dict) -> list[TextContent]:
    """Embed the query text and return ranked chunks from Qdrant.

    Args:
        arguments: Dict with a ``query`` sub-dict (DocumentEntry fields) and
                   optional ``max_results`` int.

    Returns:
        Single-element list containing a :class:`~mcp.types.TextContent` whose
        ``text`` is a JSON-encoded array of result objects.
    """
    doc_query = DocumentQuery[DocumentEntry](
        query=DocumentEntry(**arguments.get("query", {})),
        max_results=arguments.get("max_results", 5),
    )

    query_text = doc_query.query.text
    context    = doc_query.query.context
    category   = doc_query.query.category
    top_k      = doc_query.max_results

    logger.debug(
        "search_docs: text=%r context=%r category=%r top_k=%d",
        query_text, context, category, top_k,
    )

    deps = _get_deps()
    query_vector = deps.embedder.embed([query_text])[0]

    results: list[SearchResult] = deps.store.search(
        query_vector=query_vector,
        top_k=top_k,
        context=context,
        category=category,
    )

    payload = [_result_to_dict(r) for r in results]
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


async def _handle_list_categories() -> list[TextContent]:
    """Return all distinct category values stored in Qdrant."""
    logger.debug("list_categories called.")
    deps = _get_deps()
    categories = deps.store.list_categories()
    return [TextContent(type="text", text=json.dumps(categories, ensure_ascii=False))]


# ---------------------------------------------------------------------------
# MCP tool registration  (thin wrappers — schema inferred from signatures)
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_docs(
    text: str,
    context: Optional[str] = None,
    category: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """Perform semantic search over ingested PDF documents.

    Args:
        text:        Natural-language query string.
        context:     Optional. Filter results to a specific document / source
                     (matches the ``context`` payload field set during ingestion).
        category:    Optional. Filter results to a specific section or category
                     (matches the ``category`` payload field set during ingestion).
        max_results: Maximum number of chunks to return (1–100, default 5).

    Returns:
        JSON array of result objects ordered by descending relevance score.
        Each object contains: score, text, source_file, page, chunk_index,
        context, category.
    """
    response = await _handle_search_docs({
        "query": {"text": text, "context": context, "category": category},
        "max_results": max_results,
    })
    return response[0].text


@mcp.tool()
async def list_categories() -> str:
    """Return all distinct category values stored in the Qdrant collection.

    Use this to discover available document sections before filtering a
    ``search_docs`` call with a ``category`` argument.

    Returns:
        JSON array of category strings, sorted alphabetically.
    """
    response = await _handle_list_categories()
    return response[0].text


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _result_to_dict(result: SearchResult) -> dict:
    """Convert a :class:`~pdf_semantic_search.vector_store.qdrant_store.SearchResult`
    to a plain dict suitable for JSON serialisation."""
    return {
        "score":       round(result.score, 4),
        "text":        result.text,
        "source_file": result.source_file,
        "page":        result.page,
        "chunk_index": result.chunk_index,
        "context":     result.context,
        "category":    result.category,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio (blocking)."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
