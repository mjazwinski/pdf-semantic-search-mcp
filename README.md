# PDF Semantic Search — MCP Server

Local PDF semantic search using **SentenceTransformer** embeddings, **Qdrant** as the vector store, and an **MCP stdio server** to expose search to AI agents.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  AI Agent / Claude Desktop / Cursor              │
│  (MCP client)                                    │
└───────────────────┬──────────────────────────────┘
                    │ stdio (MCP protocol)
┌───────────────────▼──────────────────────────────┐
│  MCP Server  (pdf-mcp-server)                    │
│  • list_tools → ["search"]                       │
│  • call_tool  → embed query → Qdrant search      │
└───────────────────┬──────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐   ┌───────────▼───────────┐
│ SentenceTransf│   │  Qdrant (Docker)       │
│ ormer Embedder│   │  localhost:6333        │
│ (local, CPU)  │   │  collection: pdf_chunks│
└───────────────┘   └───────────────────────┘

PDF Ingestion (CLI):
  pdf-ingest file.pdf [--chunk-size 512] [--overlap 64]
  → PyMuPDFParser → chunk_text → embed → QdrantStore.upsert
```

---

## Quick Start

### 1. Install dependencies

```bash
# Requires Python ≥ 3.11 and uv
uv sync
```
alternatively:
```bash
# Requires Python ≥ 3.11 and uv
python -m uv sync
```

### 2. Start Qdrant

```bash
cp .env.example .env
docker compose up -d
```

### 3. Run the unit tests

```bash
# Fast — no Qdrant, no model download
pytest -m "not integration"

# With coverage report
pytest -m "not integration" --cov

# Full suite (requires docker compose up -d)
pytest -m integration
```

### 4. Ingest a PDF

```bash
pdf-ingest path/to/document.pdf --chunk-size 512 --overlap 64
```

### 5. Run the MCP server

See the [MCP Server Setup](#mcp-server-setup) section below for full per-platform instructions.

---

## MCP Server Setup

The server communicates over **stdio** — the MCP client (Claude Desktop, Cursor, etc.) spawns
it as a child process.  The setup differs slightly per OS because of how virtual-environment
executables are resolved.

### Prerequisites (all platforms)

1. Python ≥ 3.11 installed and on `PATH`
2. `uv` installed (`pip install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv))
3. Dependencies installed: `uv sync`
4. Qdrant running: `docker compose up -d`
5. `.env` file present: `cp .env.example .env`

---

### Windows

#### Run manually (PowerShell)

```powershell
# From the project root
.\.venv\Scripts\pdf-mcp-server.exe
```

The server writes diagnostics to `pdf-semantic-search.log` in its working
directory and also emits them to stderr. Set `DEBUG=true` in `.env` to include
debug-level messages:

```dotenv
DEBUG=true
```

#### MCP client configuration

The `command` must be the **absolute path** to the script inside `.venv` — MCP clients do not
inherit your shell's `PATH`.

```json
{
  "mcpServers": {
    "pdf-search": {
      "command": "C:\\dev\\projects\\pdf-semantic-search-mcp\\.venv\\Scripts\\pdf-mcp-server.exe",
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

> **Tip — find the exact path:**
> ```powershell
> (Get-Command pdf-mcp-server).Source
> # or, if not on PATH:
> Resolve-Path .\.venv\Scripts\pdf-mcp-server.exe
> ```

> **Cursor on Windows** — paste the config into
> `%APPDATA%\Cursor\User\globalStorage\cursor.mcp\settings.json`
> (or use *Cursor → Settings → MCP*).

---

### macOS

#### Run manually

```bash
# From the project root
.venv/bin/pdf-mcp-server
```

#### MCP client configuration

```json
{
  "mcpServers": {
    "pdf-search": {
      "command": "/Users/<you>/projects/pdf-semantic-search-mcp/.venv/bin/pdf-mcp-server",
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

> **Tip — find the exact path:**
> ```bash
> which pdf-mcp-server
> # or, if not on PATH:
> readlink -f .venv/bin/pdf-mcp-server
> ```

> **Claude Desktop on macOS** — paste the config into
> `~/Library/Application Support/Claude/claude_desktop_config.json`.

> **Cursor on macOS** — paste into
> `~/Library/Application Support/Cursor/User/globalStorage/cursor.mcp/settings.json`
> (or use *Cursor → Settings → MCP*).

> **Apple Silicon (M-series) note:** PyTorch and `sentence-transformers` ship MPS-accelerated
> wheels for arm64.  The server auto-detects MPS — no extra config needed.  If you want to
> force CPU set `"PYTORCH_ENABLE_MPS_FALLBACK": "1"` in `env` above.

---

### Linux

#### Run manually

```bash
# From the project root
.venv/bin/pdf-mcp-server
```

#### MCP client configuration

```json
{
  "mcpServers": {
    "pdf-search": {
      "command": "/home/<you>/projects/pdf-semantic-search-mcp/.venv/bin/pdf-mcp-server",
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

> **Tip — find the exact path:**
> ```bash
> which pdf-mcp-server
> # or, if not on PATH:
> realpath .venv/bin/pdf-mcp-server
> ```

> **Claude Desktop on Linux** — paste into
> `~/.config/Claude/claude_desktop_config.json`.

> **Cursor on Linux** — paste into
> `~/.config/Cursor/User/globalStorage/cursor.mcp/settings.json`
> (or use *Cursor → Settings → MCP*).

> **GPU note:** If a CUDA GPU is present, `sentence-transformers` will use it automatically.
> To pin to CPU, add `"CUDA_VISIBLE_DEVICES": ""` to `env`.

---

### Using `uv run` instead of the venv path (all platforms)

If you prefer not to hard-code the `.venv` path, use `uv run` as the command — `uv` resolves
the project environment automatically:

```json
{
  "mcpServers": {
    "pdf-search": {
      "command": "uv",
      "args": [
        "run",
        "--project", "/absolute/path/to/pdf-semantic-search-mcp",
        "pdf-mcp-server"
      ],
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

`uv` must itself be on the system `PATH` (or supply its absolute path as `command`).
On Windows `uv` is typically at `%USERPROFILE%\.cargo\bin\uv.exe` or wherever the installer placed it.

---

### Verifying the server starts correctly

Run the server manually in a terminal and send a hand-crafted MCP `initialize` request:

```bash
# macOS / Linux
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' \
  | .venv/bin/pdf-mcp-server

# Windows PowerShell
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' |
  .\.venv\Scripts\pdf-mcp-server.exe
```

A valid startup prints a JSON response containing `"result": {"protocolVersion": ...}` to stdout.
Any errors (Qdrant unreachable, model load failure) appear on stderr.

---

### 6. Configure your MCP client

Quick-reference config (replace the path for your OS — see above):

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

---

## Project Structure

```
src/pdf_semantic_search/
├── config.py                   ← Pydantic settings (env / .env)
├── models.py                   ← DocumentEntry + DocumentQuery[T] (generic)
├── embeddings/
│   └── sentence_transformer.py ← EmbeddingModel protocol + SentenceTransformer impl
├── vector_store/
│   └── qdrant_store.py         ← Qdrant upsert + search
├── pdf/
│   ├── chunker.py              ← Sliding-window text chunking
│   └── parser.py               ← PDFParserBase ABC + PyMuPDFParser default
├── ingestion/
│   ├── service.py              ← IngestionService (parse → embed → upsert)
│   └── cli.py                  ← Typer CLI (pdf-ingest entry point)
└── mcp_server/
    └── server.py               ← MCP stdio server: search_docs + list_categories
```

---

## Development Roadmap

| Step | Module | Status |
|------|--------|--------|
| 1 | Scaffold & project layout | ✅ Done |
| 2 | Config (`pydantic-settings`) | ✅ Done |
| 3 | Data models — `DocumentEntry`, `DocumentQuery` (generic) | ✅ Done |
| 4 | `SentenceTransformerEmbedder.embed()` | ✅ Done |
| 5 | `QdrantStore._connect / upsert / search` | ✅ Done |
| 6 | `chunk_text` + `PyMuPDFParser.parse` | ✅ Done |
| 7 | `IngestionService.ingest` + CLI wiring | ✅ Done |
| 8 | MCP `search_docs` + `list_categories` implementation | ✅ Done |
| 9 | Tests — conftest, e2e pipeline, coverage config | ✅ Done |
