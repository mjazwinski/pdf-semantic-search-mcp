# Copilot Instructions for PDF Semantic Search MCP

## Quick Reference

### Project Type
Python MCP server for semantic PDF search using local embeddings and Qdrant vector store.

### Python Version
Requires **Python ≥ 3.11**. Dependencies managed with `uv`.

---

## Build, Test, and Lint Commands

### Install Dependencies
```bash
uv sync
```

### Running Tests
```bash
# Unit tests only (fast — no Qdrant, no model downloads)
pytest -m "not integration"

# With coverage report (80% fail threshold in pyproject.toml)
pytest -m "not integration" --cov

# Run a single test file
pytest tests/test_models.py

# Full suite (requires `docker compose up -d`)
pytest -m integration

# Integration + slow tests
pytest -m "integration or slow"
```

### Linting
```bash
# Check with Ruff (100-char line length, src-only)
ruff check src tests

# Format code
ruff format src tests
```

### External Services
```bash
# Start Qdrant in Docker (required for integration tests and MCP server)
docker compose up -d

# Stop Qdrant
docker compose down
```

---

## High-Level Architecture

### System Diagram
```
MCP Client (Claude Desktop / Cursor)
    ↓ stdio (MCP protocol)
MCP Server (pdf-mcp-server)
    ├→ SentenceTransformer (local embeddings, CPU-friendly)
    └→ Qdrant (Docker, localhost:6333)
```

### Pipeline Workflow

1. **Ingestion CLI** (`pdf-ingest`):
   - Accepts PDF files with chunking parameters
   - Parses with PyMuPDF
   - Splits text with sliding-window chunking
   - Embeds chunks with SentenceTransformer
   - Upserts to Qdrant with payload metadata

2. **MCP Server** (`pdf-mcp-server`):
   - Exposes two tools: `search_docs` and `list_categories`
   - Lazy-loads embedding model and Qdrant client on first call
   - Returns search results with similarity scores and metadata

### Directory Structure
```
src/pdf_semantic_search/
├── config.py              ← Settings singleton (Pydantic, .env)
├── models.py              ← DocumentEntry, DocumentQuery[T] (generic)
├── embeddings/            ← EmbeddingModel protocol + SentenceTransformer
├── vector_store/          ← QdrantStore (upsert, search, list_categories)
├── pdf/                   ← PDFParserBase ABC + PyMuPDFParser, chunking
├── ingestion/             ← IngestionService, Typer CLI
└── mcp_server/            ← MCPServer (mcp >= 2.0) with stdio transport
```

---

## Key Conventions

### Configuration
- **Single source of truth**: `pdf_semantic_search/config.py` → `Settings` singleton
- Import as: `from pdf_semantic_search.config import settings`
- All config keys are snake_case, loaded from `.env` file
- Environment variables (uppercase) are automatically mapped

### Type Safety & Generics
- **DocumentQuery is generic**: `DocumentQuery[T]` where `T` extends `DocumentEntry`
- Allows extending the data model without modifying core code:
  ```python
  class RichEntry(DocumentEntry):
      author: str
  
  q = DocumentQuery[RichEntry](
      query=RichEntry(text="...", author="..."),
      max_results=5
  )
  ```

### Protocol-Based Abstraction
- **EmbeddingModel protocol** in `embeddings/`: Define your own by implementing `embed(texts: list[str]) -> list[list[float]]` and `dim` property
- **PDFParserBase ABC** in `pdf/parser.py`: Swap PDF libraries (PyMuPDF is the default)
- Allows testing without concrete implementations

### Testing Patterns
- **Test markers** (configured in `pyproject.toml`):
  - `@pytest.mark.integration` — requires live Qdrant + model download
  - `@pytest.mark.slow` — intentionally time-consuming
  - Run without external services: `pytest -m "not integration"`

- **Fixtures in `tests/conftest.py`**:
  - `make_chunks()`, `sample_chunks` — TextChunk factories
  - `make_mock_embedder()`, `mock_embedder` — Deterministic mock embeddings (dim=4, output = vector of text length)
  - `make_mock_store()`, `mock_store` — Qdrant mock that returns empty results by default
  - `make_search_result()` — SearchResult builder with sensible defaults
  - `tmp_pdf` — Temporary file with PDF magic bytes (for parser testing)

### Code Style
- **Line length**: 100 characters (Ruff configuration)
- **Coverage**: 80% minimum (enforced in CI)
- **Warnings filtered** in `pyproject.toml`:
  - DeprecationWarnings from Pydantic/PyTorch
  - UserWarnings from sentence-transformers

### MCP Server Development

#### Server Implementation
- Uses **MCPServer** from `mcp >= 2.0` (not the older `FastMCP` from mcp 1.x)
- Tool schemas inferred from function signatures (no manual `inputSchema`)
- Lazy initialization: embedding model and Qdrant client created once on first tool call
- stdio transport (default for local/desktop MCP clients)

#### Exposed Tools
- `search_docs(text, context?, category?, max_results?)` → JSON array of SearchResult objects
- `list_categories()` → JSON array of distinct category strings

#### Testing the Server with MCP Clients

**Manual server startup:**
```bash
# Start the server in stdio mode (sends/receives JSON-RPC over stdin/stdout)
pdf-mcp-server
```

**Test with an MCP initialize request:**
```bash
# macOS / Linux
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' \
  | .venv/bin/pdf-mcp-server

# Windows PowerShell
'{"jsonrpc":"2.0","id":1,"method":"initialize",...}' | .\.venv\Scripts\pdf-mcp-server.exe
```
Expected: JSON response with `"result": {"protocolVersion": "..."}`. Errors appear on stderr.

**Client configuration** (Claude Desktop / Cursor):
```json
{
  "mcpServers": {
    "pdf-search": {
      "command": "<absolute-path-to-pdf-mcp-server>",
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "QDRANT_COLLECTION": "pdf_chunks",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

#### Key MCP Concepts Used Here
- **Tool definition**: `@mcp.tool()` decorator on `search_docs` and `list_categories`
- **Type annotations**: Parameter types and return types drive JSON schema generation
- **Optional parameters**: `Optional[str]` becomes `"required": false` in schema
- **Pydantic integration**: Complex types (e.g., list of SearchResult) are serialized to JSON via `.model_dump()` or `.model_dump_json()`

### Dependencies
- **PDF**: PyMuPDF (fitz)
- **Embeddings**: sentence-transformers (SentenceTransformer)
- **Vector Store**: qdrant-client
- **MCP**: mcp >= 2.0
- **Config**: pydantic-settings
- **CLI**: Typer + Rich (for colored output)
- **Testing**: pytest, pytest-cov

---

## Environment Variables
See `.env.example` for defaults. Key variables:
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION` — Vector store connection
- `EMBEDDING_MODEL` — SentenceTransformer model name (defaults to `all-MiniLM-L6-v2`)
- `EMBEDDING_DIM` — Vector dimension (must match model output; 384 for all-MiniLM-L6-v2)
- `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` — PDF chunking defaults
