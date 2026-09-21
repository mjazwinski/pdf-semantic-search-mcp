"""MCP server — exposes ``search_docs`` and ``list_categories`` over stdio.

Start with:
    pdf-mcp-server
    # or:
    python -m pdf_semantic_search.mcp_server.server

MCP client configuration (e.g. Claude Desktop / Cursor):
    {
      "mcpServers": {
        "pdf-search": {
          "command": "pdf-mcp-server"
        }
      }
    }

Exposed tools
-------------
search_docs(doc_query)
    Semantic search over ingested PDF chunks.
    ``doc_query`` is a JSON object matching :class:`~pdf_semantic_search.models.DocumentQuery`.
    ``context`` and ``category`` fields inside ``query`` are forwarded as Qdrant
    payload filters so only matching chunks are returned.

    Returns a JSON array of result objects, each containing:
        score, text, source_file, page, chunk_index, context, category

list_categories()
    Returns a JSON array of all distinct ``category`` strings stored in Qdrant.
    Useful for letting an agent enumerate document sections before narrowing a search.

Dependency lifecycle
--------------------
Both the embedding model and the Qdrant client are expensive to initialise.
They are created once on the first tool call via :func:`_get_deps` and reused
for the lifetime of the server process.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import mcp.server.stdio
from mcp.server import Server
from mcp.types import TextContent, Tool

from pdf_semantic_search.config import settings
from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder
from pdf_semantic_search.models import DocumentEntry, DocumentQuery
from pdf_semantic_search.vector_store.qdrant_store import QdrantStore, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = Server("pdf-semantic-search")


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
            device=None,   # auto-detect CPU / CUDA / MPS
            normalize=True,
        )
        # Trigger model load now so the first search call isn't slow
        _ = embedder.dim

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            dim=embedder.dim,
        )
        store._connect()

        _deps = _Deps(embedder=embedder, store=store)
        logger.info("Dependencies ready (model=%r, dim=%d).", settings.embedding_model, embedder.dim)

    return _deps


def _reset_deps() -> None:
    """Replace the singleton with *None*.  Used in tests only."""
    global _deps
    _deps = None


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise available tools to the MCP client."""
    return [
        Tool(
            name="search_docs",
            description=(
                "Perform semantic search over ingested PDF documents. "
                "Pass a query object with a 'text' field (required) and optional "
                "'context' / 'category' fields to narrow results to a specific document "
                "or section. Returns a JSON array of matching chunks with relevance scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "Search parameters (DocumentEntry).",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Natural-language query string.",
                            },
                            "context": {
                                "type": "string",
                                "description": (
                                    "Optional. Filter results to a specific document / source "
                                    "(matches the 'context' payload field set during ingestion)."
                                ),
                            },
                            "category": {
                                "type": "string",
                                "description": (
                                    "Optional. Filter results to a specific section or category "
                                    "(matches the 'category' payload field set during ingestion)."
                                ),
                            },
                        },
                        "required": ["text"],
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return (1–100, default 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_categories",
            description=(
                "Return all distinct 'category' values stored in the Qdrant collection. "
                "Use this to discover available document sections before filtering a search."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route incoming tool calls to the appropriate handler."""
    if name == "search_docs":
        return await _handle_search_docs(arguments)
    if name == "list_categories":
        return await _handle_list_categories()
    raise ValueError(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_search_docs(arguments: dict) -> list[TextContent]:
    """Embed the query text and return ranked chunks from Qdrant.

    Args:
        arguments: Raw MCP tool arguments dict containing ``query`` (object)
                   and optional ``max_results`` (int).

    Returns:
        A single :class:`~mcp.types.TextContent` whose ``text`` is a
        JSON-encoded array of result objects, ordered by descending score.

    Raises:
        ValueError:  If ``arguments["query"]`` is missing or invalid.
        RuntimeError: If the embedding model or Qdrant is unreachable.
    """
    # ── 1. Parse & validate input ─────────────────────────────────────────────
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

    # ── 2. Embed the query ────────────────────────────────────────────────────
    deps = _get_deps()
    query_vector = deps.embedder.embed([query_text])[0]

    # ── 3. Search Qdrant (with optional filters) ──────────────────────────────
    results: list[SearchResult] = deps.store.search(
        query_vector=query_vector,
        top_k=top_k,
        context=context,
        category=category,
    )

    # ── 4. Serialise to JSON ──────────────────────────────────────────────────
    payload = [_result_to_dict(r) for r in results]
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


async def _handle_list_categories() -> list[TextContent]:
    """Return all distinct category values stored in Qdrant.

    Returns:
        A single :class:`~mcp.types.TextContent` whose ``text`` is a
        JSON-encoded sorted list of category strings.
    """
    logger.debug("list_categories called.")

    deps = _get_deps()
    categories = deps.store.list_categories()

    return [TextContent(type="text", text=json.dumps(categories, ensure_ascii=False))]


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _result_to_dict(result: SearchResult) -> dict:
    """Convert a :class:`~pdf_semantic_search.vector_store.qdrant_store.SearchResult`
    to a plain dict suitable for JSON serialisation.

    The ``score`` is rounded to 4 decimal places to keep responses concise.
    """
    return {
        "score":        round(result.score, 4),
        "text":         result.text,
        "source_file":  result.source_file,
        "page":         result.page,
        "chunk_index":  result.chunk_index,
        "context":      result.context,
        "category":     result.category,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio (blocking)."""
    import asyncio

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_serve())


async def _serve() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    main()
