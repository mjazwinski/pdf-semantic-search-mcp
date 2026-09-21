"""CLI entry point for PDF ingestion.

Usage
-----
    # Via installed script
    pdf-ingest path/to/document.pdf

    # With options
    pdf-ingest report.pdf --chunk-size 256 --overlap 32 --context "Q4 Report" --category "finance"

    # Directly
    python -m pdf_semantic_search.ingestion.cli path/to/document.pdf
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pdf_semantic_search.config import settings
from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder
from pdf_semantic_search.ingestion.service import IngestionService
from pdf_semantic_search.pdf.parser import PyMuPDFParser
from pdf_semantic_search.vector_store.qdrant_store import QdrantStore

app = typer.Typer(
    help="Ingest a PDF file into the Qdrant vector store.",
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(
        ...,
        help="Path to the PDF file to ingest.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    chunk_size: int = typer.Option(
        None,
        "--chunk-size", "-c",
        help="Characters per chunk. Defaults to DEFAULT_CHUNK_SIZE in config.",
    ),
    overlap: int = typer.Option(
        None,
        "--overlap", "-o",
        help="Character overlap between chunks. Defaults to DEFAULT_CHUNK_OVERLAP in config.",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size", "-b",
        help="Number of chunks to embed and upsert per batch.",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        help=(
            "Label stored in every chunk's 'context' payload field. "
            "Used as a Qdrant filter in search (e.g. document name or title). "
            "Defaults to the PDF filename."
        ),
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help=(
            "Label stored in every chunk's 'category' payload field. "
            "Used as a Qdrant filter in search (e.g. document section or topic)."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable DEBUG logging.",
    ),
) -> None:
    """Parse *PDF_PATH* and upsert embeddings into the Qdrant vector store."""

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Resolve defaults ──────────────────────────────────────────────────────
    chunk_size = chunk_size or settings.default_chunk_size
    overlap = overlap or settings.default_chunk_overlap
    context = context or pdf_path.name   # default context = filename

    # ── Banner ────────────────────────────────────────────────────────────────
    console.rule("[bold blue]PDF Semantic Search — Ingestion")
    console.print(f"  [bold]File:[/bold]       {pdf_path}")
    console.print(f"  [bold]Chunk size:[/bold] {chunk_size} chars  |  [bold]Overlap:[/bold] {overlap} chars")
    console.print(f"  [bold]Batch size:[/bold] {batch_size}")
    console.print(f"  [bold]Context:[/bold]    {context!r}")
    console.print(f"  [bold]Category:[/bold]   {category!r}")
    console.print(
        f"  [bold]Qdrant:[/bold]     {settings.qdrant_host}:{settings.qdrant_port}  "
        f"collection=[cyan]{settings.qdrant_collection}[/cyan]"
    )
    console.print(
        f"  [bold]Model:[/bold]      [cyan]{settings.embedding_model}[/cyan]"
    )
    console.print()

    # ── Wire dependencies ─────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:

        # 1. Load model
        task_model = progress.add_task("Loading embedding model …", total=1)
        try:
            embedder = SentenceTransformerEmbedder(
                model_name=settings.embedding_model,
                device=None,  # auto-detect CPU / CUDA / MPS
            )
            _ = embedder.dim  # trigger lazy load now so progress bar is accurate
        except Exception as exc:
            console.print(f"[red]✗ Failed to load embedding model:[/red] {exc}")
            raise typer.Exit(code=1)
        progress.update(task_model, advance=1, description="Embedding model loaded ✓")

        # 2. Connect to Qdrant
        task_qdrant = progress.add_task("Connecting to Qdrant …", total=1)
        try:
            store = QdrantStore(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection=settings.qdrant_collection,
                dim=embedder.dim,
            )
            store._connect()
        except Exception as exc:
            console.print(f"[red]✗ Could not connect to Qdrant:[/red] {exc}")
            console.print(
                f"  Is Docker running?  "
                f"Try: [bold]docker compose up -d[/bold]"
            )
            raise typer.Exit(code=1)
        progress.update(task_qdrant, advance=1, description="Qdrant connected ✓")

        # 3. Parse + embed + upsert
        task_ingest = progress.add_task("Ingesting …", total=None)  # indeterminate until parsed
        t0 = time.perf_counter()
        try:
            service = IngestionService(
                parser=PyMuPDFParser(),
                embedder=embedder,
                store=store,
            )
            total = service.ingest(
                pdf_path=pdf_path,
                chunk_size=chunk_size,
                overlap=overlap,
                batch_size=batch_size,
                context=context,
                category=category,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]✗ File not found:[/red] {exc}")
            raise typer.Exit(code=1)
        except Exception as exc:
            console.print(f"[red]✗ Ingestion failed:[/red] {exc}")
            raise typer.Exit(code=1)

        elapsed = time.perf_counter() - t0
        progress.update(task_ingest, completed=1, total=1, description="Ingestion complete ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Done")
    console.print(
        f"  [green]✓[/green] Ingested [bold]{total}[/bold] chunk(s) "
        f"in [bold]{elapsed:.1f}s[/bold]"
    )
    console.print(
        f"  Collection [cyan]{settings.qdrant_collection}[/cyan] "
        f"@ {settings.qdrant_host}:{settings.qdrant_port}"
    )


if __name__ == "__main__":
    app()
