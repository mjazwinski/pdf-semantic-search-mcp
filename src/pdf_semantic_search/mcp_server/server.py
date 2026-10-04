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
search_docs(text, category?, max_results?)
    Semantic search over ingested PDF chunks.
    ``category`` is optional Qdrant payload filters.
    Returns a JSON array of result objects:
        [{score, text, source_file, page, chunk_index, category}, ...]

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
from pathlib import Path
from typing import Optional

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import TextContent

from pdf_semantic_search.config import settings
from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder
from pdf_semantic_search.embeddings.sparse_embedder import FastEmbedSparseEmbedder
from pdf_semantic_search.models import DocumentEntry, DocumentQuery
from pdf_semantic_search.vector_store.qdrant_store import QdrantStore, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCPServer application  (mcp 2.x — formerly FastMCP in mcp 1.x)
# ---------------------------------------------------------------------------

mcp = MCPServer(
    "pdf-semantic-search",
    instructions=(
        "Search ingested PDF documents by natural-language query. "
        "Use list_categories to discover filterable sections, then "
        "search_docs with an optional category to narrow results."
    ),
)
# ---------------------------------------------------------------------------
# Lazy-initialised dependencies (created once, reused for server lifetime)
# ---------------------------------------------------------------------------


@dataclass
class _Deps:
    embedder: SentenceTransformerEmbedder
    sparse_embedder: FastEmbedSparseEmbedder
    store: QdrantStore


_deps: Optional[_Deps] = None  # module-level singleton


def _get_deps() -> _Deps:
    """Return the shared embedder + sparse embedder + store, initialising on first call."""
    global _deps
    if _deps is None:
        logger.info("Initialising MCP server dependencies …")

        embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            device=None,    # auto-detect CPU / CUDA / MPS
            normalize=True,
        )
        _ = embedder.dim  # trigger dense model load now

        sparse_embedder = FastEmbedSparseEmbedder(
            model_name=settings.sparse_embedding_model,
        )
        # Warm up the sparse model so first search is not slow
        sparse_embedder.embed_sparse(["warmup"])

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            dim=embedder.dim,
            dense_vector_name=settings.dense_vector_name,
            sparse_vector_name=settings.sparse_vector_name,
        )
        store._connect()

        _deps = _Deps(
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            store=store,
        )
        logger.info(
            "Dependencies ready (dense=%r dim=%d, sparse=%r).",
            settings.embedding_model,
            embedder.dim,
            settings.sparse_embedding_model,
        )

    return _deps


def _reset_deps() -> None:
    """Replace the singleton with *None*.  Used in tests only."""
    global _deps
    _deps = None


# ---------------------------------------------------------------------------
# Tool handlers  (pure business logic — testable without MCPServer)
# ---------------------------------------------------------------------------


async def _handle_search_docs(arguments: dict, 
        mcpContext: Context,) -> list[TextContent]:
    """Embed the query text and return ranked chunks from Qdrant.

    Args:
        arguments: Dict with a ``query`` sub-dict (DocumentEntry fields) and
                   optional ``max_results`` int.

    Returns:
        Single-element list containing a :class:`~mcp.types.TextContent` whose
        ``text`` is a JSON-encoded array of result objects.
    """
    await mcpContext.report_progress(0, 100, "Building query")
    doc_query = DocumentQuery[DocumentEntry](
        query=DocumentEntry(**arguments.get("query", {})),
        max_results=arguments.get("max_results", 5),
    )

    query_text = doc_query.query.text
    category   = doc_query.query.category
    version    = doc_query.query.version
    top_r      = doc_query.max_results

    logger.debug(
        "search_docs: text=%r category=%r version=%r top_r=%d",
        query_text, category, version, top_r,
    )
    await mcpContext.report_progress(20, 100, "Getting deps")
    deps = _get_deps()
    await mcpContext.report_progress(30, 100, "Got deps — embedding query")
    query_dense = deps.embedder.embed([query_text])[0]
    query_sparse = deps.sparse_embedder.embed_sparse([query_text])[0]
    await mcpContext.report_progress(40, 100, "Executing hybrid query (dense + sparse RRF)")
    results: list[SearchResult] = deps.store.search(
        query_dense=query_dense,
        query_sparse=query_sparse,
        top_r=top_r,
        category=category,
        version=version,
    )

    await mcpContext.report_progress(90, 100, "Processing result")
    payload = [_result_to_dict(r) for r in results]
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


async def _handle_list_categories() -> list[TextContent]:
    """Return all distinct category values stored in Qdrant."""
    logger.debug("list_categories called.")
    deps = _get_deps()
    categories = deps.store.list_categories()
    return [TextContent(type="text", text=json.dumps(categories, ensure_ascii=False))]


# --------------------------
# MCP resource registration 
# --------------------------

@mcp.resource("usage://extended_description")
def get_config() -> str:
    """Extended description of this MCP for less capabable harnesses"""
    return """pdf-semantic-search-mcp provides access to data in categories you can get by calling list_categories tool. 
    If one of categories matches current subject you can serch for terms or sentences (eg. car engine or what car engines can I put in VW Golf)
    To execute search use search_docs tool passing category and search term/sentence
    """

# ---------------------------------------------------------------------------
# MCP tool registration  (thin wrappers — schema inferred from signatures)
# ---------------------------------------------------------------------------


@mcp.tool(name="search_docs",
          description="Allows to search for information in one of categories listed by the call to list_categories tool.")
async def search_docs(
    text: str,
    mcpContext: Context,
    category: Optional[str] = None,
    version: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """Perform semantic search over ingested PDF documents.

    Args:
        text:        Natural-language query string.
        category:    Optional. Filter results to a specific section or category
                     (matches the ``category`` payload field set during ingestion).
        version:     Optional. Filter results to a specific document version
                     (matches the ``version`` payload field set during ingestion).
        max_results: Maximum number of chunks to return (1–100, default 5).

    Returns:
        JSON array of result objects ordered by descending relevance score.
        Each object contains: score, text, source_file, page, chunk_index,
        category, version.
    """
    try:
        response = await _handle_search_docs({
            "query": {"text": text, "category": category, "version": version},
            "max_results": max_results,
        }, mcpContext)
        return response[0].text
    except Exception as e:
        logger.exception(f"search_docs failed {e}")
        raise


@mcp.tool(name="list_categories",
          description="Returns list of semantic categories that can be searched e.g. training, artificial intelligence, motorization.")
async def list_categories(mcpContext: Context) -> str:
    """Return all distinct category values stored in the Qdrant collection.

    Use this to discover available document sections before filtering a
    ``search_docs`` call with a ``category`` argument.

    Returns:
        JSON array of category strings, sorted alphabetically.
    """
    logger.debug(f"Handling request {mcpContext.request_id} for list_categories")
    try:
        response = await _handle_list_categories()
        return response[0].text
    except Exception:
        logger.exception("list_categories failed")
        raise


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
        "category":    result.category,
        "version":     result.version,
    }


def _disable_model_progress_bars() -> None:
    """Disable model-loading progress output that can corrupt MCP stdio."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio (blocking)."""
    _disable_model_progress_bars()
    log_level = logging.DEBUG if settings.debug else logging.INFO
    log_path = Path(settings.log_path) if settings.log_path else Path.cwd() / "pdf-semantic-search.log"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("Starting MCP server (debug=%s, log_file=%s).", settings.debug, log_path)
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.exception(f"MCP server stopped with an error: {e}")
        raise


if __name__ == "__main__":
    main()
