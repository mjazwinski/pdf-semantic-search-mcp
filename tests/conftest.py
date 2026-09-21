"""Shared pytest fixtures and configuration for the test suite.

Fixtures defined here are available to every test file without importing.

Marker summary
--------------
``integration``
    Requires live external services (running Qdrant container, internet
    access for model download).  Skipped by default in CI.
    Run with: ``pytest -m integration``

``slow``
    Tests that are intentionally slow (e.g. large-model inference).
    Run with: ``pytest -m slow``

Deselect all slow/integration tests:
    ``pytest -m "not integration and not slow"``
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdf_semantic_search.pdf.parser import TextChunk
from pdf_semantic_search.vector_store.qdrant_store import SearchResult


# ---------------------------------------------------------------------------
# File-system fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_pdf(tmp_path: Path) -> Path:
    """A temporary file that passes the ``Path.exists()`` check used by parsers.

    The file contains a minimal PDF header so tools that peek at magic bytes
    do not immediately reject it.  Real PyMuPDF parsing is still mocked in
    unit tests.
    """
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return p


# ---------------------------------------------------------------------------
# TextChunk factories
# ---------------------------------------------------------------------------


def make_chunks(
    n: int,
    source_file: str = "doc.pdf",
    page: int = 0,
    text_prefix: str = "chunk text",
) -> list[TextChunk]:
    """Return *n* :class:`TextChunk` objects with sequential chunk indices."""
    return [
        TextChunk(
            text=f"{text_prefix} {i}",
            source_file=source_file,
            page=page,
            chunk_index=i,
        )
        for i in range(n)
    ]


@pytest.fixture()
def sample_chunks() -> list[TextChunk]:
    """Five simple chunks from a single page — ready for ingestion tests."""
    return make_chunks(5)


# ---------------------------------------------------------------------------
# Mock embedding model
# ---------------------------------------------------------------------------


def make_mock_embedder(dim: int = 4) -> MagicMock:
    """Return a MagicMock that satisfies the EmbeddingModel protocol."""
    embedder = MagicMock()
    embedder.dim = dim
    embedder.embed.side_effect = lambda texts: [[float(len(t))] * dim for t in texts]
    return embedder


@pytest.fixture()
def mock_embedder() -> MagicMock:
    """A dim=4 mock embedder whose vectors are deterministic on text length."""
    return make_mock_embedder(dim=4)


# ---------------------------------------------------------------------------
# Mock Qdrant store
# ---------------------------------------------------------------------------


def make_mock_store(
    search_results: list[SearchResult] | None = None,
    categories: list[str] | None = None,
) -> MagicMock:
    """Return a MagicMock that satisfies the QdrantStore interface."""
    store = MagicMock()
    store.search.return_value = search_results or []
    store.list_categories.return_value = categories or []
    return store


@pytest.fixture()
def mock_store() -> MagicMock:
    """A QdrantStore mock that returns empty search results by default."""
    return make_mock_store()


# ---------------------------------------------------------------------------
# SearchResult factory
# ---------------------------------------------------------------------------


def make_search_result(
    score: float = 0.90,
    text: str = "Sample chunk text.",
    source_file: str = "report.pdf",
    page: int = 0,
    chunk_index: int = 0,
    context: str | None = "report.pdf",
    category: str | None = "intro",
) -> SearchResult:
    """Build a :class:`SearchResult` with sensible defaults."""
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
