# PDF Semantic Search — MCP Server

Local PDF semantic search using **hybrid dense + sparse embeddings**, **Qdrant** as the vector
store, and an **MCP stdio server** to expose search to AI agents.

Every search combines two complementary retrieval strategies — semantic similarity and exact
term matching — fused by Reciprocal Rank Fusion (RRF) into a single ranked result list.

---

## How Search Works

### Two retrieval strategies, one fused result

```
Query: "how to tell that engine is hot"
            │
  ┌─────────┴──────────┐
  │                    │
  ▼                    ▼
Dense ANN           Sparse (SPLADE++)
SentenceTransformer prithivida/Splade_PP_en_v1
"engine is hot"     "engine"  "hot"  "coolant"
  │  →  cosine        │  →  term weights
  │                   │
  └──────┬────────────┘
         ▼
  Reciprocal Rank Fusion (RRF)
  Re-ranks by position in both lists
         │
         ▼
  Top-k results  ←  best of both worlds
```

| Strategy | Model | What it finds |
|---|---|---|
| **Dense (semantic)** | `all-MiniLM-L6-v2` (SentenceTransformer) | Conceptually related passages — "engine warmup" for query "how to tell that engine is hot" |
| **Sparse (term-level)** | `prithivida/Splade_PP_en_v1` (SPLADE++) | Exact or near-exact keyword matches — "hot engine" or "coolant temperature" |
| **RRF fusion** | Qdrant built-in | Surfaces documents that rank highly in *either or both* legs |

Both models run **entirely locally** — no API key, no cloud calls.
Dense vectors are downloaded once by `sentence-transformers`; sparse vectors are downloaded once
by `fastembed` (ONNX runtime, ~500 MB, cached in `~/.cache/fastembed/`).

### Why this matters

| Query type | Dense alone | Sparse alone | Hybrid (RRF) |
|---|---|---|---|
| Natural language: *"how to tell that engine is hot"* | ✅ good | ⚠ misses — no exact terms | ✅ best |
| Exact keyword: *"hot engine"* | ⚠ may miss paraphrases | ✅ good | ✅ best |
| Mixed: *"engine temperature warning"* | ✅ good | ✅ good | ✅ best |
| Rare technical term not in dense vocabulary | ❌ poor | ✅ good (SPLADE lexical) | ✅ recovers |

### Qdrant collection layout

Each ingested chunk is stored as a single Qdrant point with **two named vectors**:

```
Point {
  id:      <deterministic UUID from SHA-1(source:page:chunk_index)>
  vector: {
    "dense":  [0.12, -0.03, …]   # 384-dim float (SentenceTransformer, cosine)
    "sparse": {3: 0.9, 42: 0.4, …}  # non-zero SPLADE++ weights (vocab indices)
  }
  payload: {
    "text":        "…chunk text…",
    "source_file": "manual.pdf",
    "page":        4,
    "chunk_index": 2,
    "category":    "maintenance",   # optional, filterable
    "version":     "v2",            # optional, filterable
  }
}
```

> ⚠ **Upgrading from a previous version** — if you have an existing Qdrant collection created
> before sparse vector support was added, you must delete and recreate it:
>
> ```bash
> docker compose down -v
> docker compose up -d
> # then re-ingest your PDFs
> ```

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
│  • search_docs  → embed query (dense + sparse)   │
│                 → Qdrant hybrid search (RRF)     │
│  • list_categories → scroll payload              │
└─────────┬───────────────────────┬────────────────┘
          │                       │
┌─────────▼──────────┐  ┌─────────▼──────────────┐
│ Dense Embedder     │  │ Sparse Embedder         │
│ SentenceTransformer│  │ SPLADE++ via fastembed  │
│ all-MiniLM-L6-v2  │  │ prithivida/Splade_PP_en │
│ (384-dim, local)   │  │ (ONNX, local)           │
└────────────────────┘  └─────────────────────────┘
                                │
                    ┌───────────▼───────────────────┐
                    │  Qdrant (Docker)               │
                    │  localhost:6333                │
                    │  collection: pdf_chunks        │
                    │  • named vector "dense"        │
                    │  • named vector "sparse"       │
                    └───────────────────────────────┘

PDF Ingestion (CLI):
  pdf-ingest file.pdf [--chunk-size 512] [--overlap 64] [--category "…"] [--version "v2"]
  → PyMuPDFParser → chunk_text
      → SentenceTransformerEmbedder.embed()   (dense)
      → FastEmbedSparseEmbedder.embed_sparse() (sparse)
      → QdrantStore.upsert()
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

# Full suite (requires docker compose up -d + model download)
pytest -m integration
```

### 4. Ingest a PDF

```bash
pdf-ingest path/to/document.pdf

# With optional metadata for filtering
pdf-ingest manual.pdf --category "maintenance" --version "v2" --chunk-size 512 --overlap 64
```

On first run the sparse model (`prithivida/Splade_PP_en_v1`) is downloaded once and cached in
`~/.cache/fastembed/`.  Subsequent ingestions start immediately.

### 5. Run the MCP server

See the [MCP Server Setup](#mcp-server-setup) section below for full per-platform instructions.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and adjust as needed:

```dotenv
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=pdf_chunks

# Dense embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
DENSE_VECTOR_NAME=dense          # named vector field in Qdrant

# Sparse embeddings (SPLADE++ via fastembed — runs locally)
SPARSE_EMBEDDING_MODEL=prithivida/Splade_PP_en_v1
SPARSE_VECTOR_NAME=sparse        # named vector field in Qdrant

# Ingestion defaults
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=64
```

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
directory by default and also emits them to stderr.  Set `DEBUG=true` in `.env`
to include debug-level messages.  Set `LOG_PATH` to override the log file path:

```dotenv
DEBUG=true
LOG_PATH=C:\path\to\pdf-semantic-search.log
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
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "SPARSE_EMBEDDING_MODEL": "prithivida/Splade_PP_en_v1"
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
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "SPARSE_EMBEDDING_MODEL": "prithivida/Splade_PP_en_v1"
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
> wheels for arm64.  The server auto-detects MPS — no extra config needed.  The sparse
> embedder uses ONNX (CPU) regardless of the dense device setting.

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
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "SPARSE_EMBEDDING_MODEL": "prithivida/Splade_PP_en_v1"
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

> **GPU note:** If a CUDA GPU is present, `sentence-transformers` will use it automatically
> for dense embeddings.  To pin to CPU, add `"CUDA_VISIBLE_DEVICES": ""` to `env`.  The
> sparse embedder always uses ONNX (CPU).

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
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "SPARSE_EMBEDDING_MODEL": "prithivida/Splade_PP_en_v1"
      }
    }
  }
}
```

`uv` must itself be on the system `PATH` (or supply its absolute path as `command`).
On Windows `uv` is typically at `%USERPROFILE%\.cargo\bin\uv.exe` or wherever the installer
placed it.

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

## Project Structure

```
src/pdf_semantic_search/
├── config.py                        ← Pydantic settings (env / .env)
├── models.py                        ← DocumentEntry, DocumentQuery[T], SparseEmbedding
├── embeddings/
│   ├── sentence_transformer.py      ← EmbeddingModel protocol + SentenceTransformer impl
│   └── sparse_embedder.py           ← SparseEmbeddingModel protocol + FastEmbedSparseEmbedder
├── vector_store/
│   └── qdrant_store.py              ← Qdrant upsert (dense+sparse) + hybrid RRF search
├── pdf/
│   ├── chunker.py                   ← Sliding-window text chunking
│   └── parser.py                    ← PDFParserBase ABC + PyMuPDFParser
├── ingestion/
│   ├── service.py                   ← IngestionService (parse → embed dense+sparse → upsert)
│   └── cli.py                       ← Typer CLI (pdf-ingest entry point)
└── mcp_server/
    └── server.py                    ← MCP stdio server: search_docs + list_categories
```

---

## Development Roadmap

| Step | Module | Status |
|------|--------|--------|
| 1 | Scaffold & project layout | ✅ Done |
| 2 | Config (`pydantic-settings`) | ✅ Done |
| 3 | Data models — `DocumentEntry`, `DocumentQuery` (generic), `SparseEmbedding` | ✅ Done |
| 4 | `SentenceTransformerEmbedder.embed()` (dense) | ✅ Done |
| 5 | `QdrantStore._connect / upsert / search` | ✅ Done |
| 6 | `chunk_text` + `PyMuPDFParser.parse` | ✅ Done |
| 7 | `IngestionService.ingest` + CLI wiring | ✅ Done |
| 8 | MCP `search_docs` + `list_categories` implementation | ✅ Done |
| 9 | Tests — conftest, e2e pipeline, coverage config | ✅ Done |
| 10 | Sparse embeddings — `FastEmbedSparseEmbedder` (SPLADE++) | ✅ Done |
| 11 | Hybrid search — Qdrant Prefetch + RRF fusion | ✅ Done |
