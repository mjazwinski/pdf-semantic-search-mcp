"""Tests for PDFParserBase and PyMuPDFParser.

Unit tests mock PyMuPDF (fitz) so no real PDF file is needed.
Integration tests use a real (tiny) PDF and are gated behind
``@pytest.mark.integration``.
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pdf_semantic_search.pdf.parser import PDFParserBase, PyMuPDFParser, TextChunk


# ---------------------------------------------------------------------------
# Helpers — build a fake fitz module
# ---------------------------------------------------------------------------


def _make_fitz_page(text: str) -> MagicMock:
    """Return a MagicMock that behaves like a fitz.Page."""
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_fitz_doc(pages: list[str]) -> MagicMock:
    """Return a MagicMock that behaves like a fitz.Document."""
    doc = MagicMock()
    doc.__len__ = lambda self: len(pages)
    doc.__getitem__ = lambda self, i: _make_fitz_page(pages[i])
    # Support context manager
    doc.__enter__ = lambda self: self
    doc.__exit__ = MagicMock(return_value=False)
    return doc


def _patch_fitz(pages: list[str]):
    """Patch fitz.open to return a fake document with *pages*."""
    fake_doc = _make_fitz_doc(pages)
    return patch("fitz.open", return_value=fake_doc)


# ---------------------------------------------------------------------------
# PDFParserBase — abstract interface
# ---------------------------------------------------------------------------


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        PDFParserBase()  # type: ignore[abstract]


def test_subclass_must_implement_parse():
    class Incomplete(PDFParserBase):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_subclass_is_accepted():
    class Concrete(PDFParserBase):
        def parse(self, pdf_path, chunk_size=512, overlap=64):
            return []

    parser = Concrete()
    assert isinstance(parser, PDFParserBase)


# ---------------------------------------------------------------------------
# TextChunk dataclass
# ---------------------------------------------------------------------------


def test_text_chunk_fields():
    chunk = TextChunk(text="hello", source_file="doc.pdf", page=0, chunk_index=2)
    assert chunk.text == "hello"
    assert chunk.source_file == "doc.pdf"
    assert chunk.page == 0
    assert chunk.chunk_index == 2


# ---------------------------------------------------------------------------
# PyMuPDFParser — unit tests (fitz mocked)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_pdf(tmp_path: Path) -> Path:
    """Create a zero-byte file that satisfies the existence check."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"")
    return p


def test_parse_raises_file_not_found():
    parser = PyMuPDFParser()
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        parser.parse(Path("/nonexistent/file.pdf"))


def test_parse_returns_list_of_text_chunks(tmp_pdf):
    pages = ["Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu."]
    with _patch_fitz(pages):
        parser = PyMuPDFParser()
        chunks = parser.parse(tmp_pdf, chunk_size=512, overlap=64)

    assert isinstance(chunks, list)
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_parse_single_page_short_text(tmp_pdf):
    """A page shorter than chunk_size must produce exactly one chunk."""
    pages = ["Short text."]
    with _patch_fitz(pages):
        chunks = PyMuPDFParser().parse(tmp_pdf, chunk_size=512, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
    assert chunks[0].page == 0
    assert chunks[0].chunk_index == 0


def test_parse_source_file_is_filename_not_full_path(tmp_pdf):
    pages = ["Some content here."]
    with _patch_fitz(pages):
        chunks = PyMuPDFParser().parse(tmp_pdf, chunk_size=512, overlap=0)

    assert chunks[0].source_file == tmp_pdf.name  # just "sample.pdf"
    assert "/" not in chunks[0].source_file
    assert "\\" not in chunks[0].source_file


def test_parse_empty_page_produces_no_chunks(tmp_pdf):
    pages = ["   \n\n   "]  # whitespace only
    with _patch_fitz(pages):
        chunks = PyMuPDFParser().parse(tmp_pdf, chunk_size=512, overlap=0)

    assert chunks == []


def test_parse_multi_page(tmp_pdf):
    pages = ["Page one content.", "Page two content.", "Page three content."]
    with _patch_fitz(pages):
        chunks = PyMuPDFParser().parse(tmp_pdf, chunk_size=512, overlap=0)

    assert len(chunks) == 3
    assert chunks[0].page == 0
    assert chunks[1].page == 1
    assert chunks[2].page == 2


def test_parse_chunk_index_resets_per_page(tmp_pdf):
    long_text = "word " * 300  # forces multiple chunks per page
    pages = [long_text, long_text]
    with _patch_fitz(pages):
        chunks = PyMuPDFParser().parse(tmp_pdf, chunk_size=100, overlap=10)

    page0_chunks = [c for c in chunks if c.page == 0]
    page1_chunks = [c for c in chunks if c.page == 1]

    # chunk_index must start at 0 on each page
    assert page0_chunks[0].chunk_index == 0
    assert page1_chunks[0].chunk_index == 0
    # and be consecutive
    for i, c in enumerate(page0_chunks):
        assert c.chunk_index == i


def test_parse_passes_chunk_size_and_overlap(tmp_pdf):
    """chunk_text must be called with the exact parameters given to parse()."""
    pages = ["word " * 200]
    with _patch_fitz(pages):
        with patch(
            "pdf_semantic_search.pdf.parser.chunk_text", wraps=lambda t, **kw: [t[:50]]
        ) as mock_chunk:
            PyMuPDFParser().parse(tmp_pdf, chunk_size=99, overlap=13)

    mock_chunk.assert_called_once_with(pages[0].strip(), chunk_size=99, overlap=13)


def test_parse_skips_truly_empty_pages(tmp_pdf):
    pages = ["real content here", "", "more content"]
    with _patch_fitz(pages):
        chunks = PyMuPDFParser().parse(tmp_pdf, chunk_size=512, overlap=0)

    pages_with_chunks = {c.page for c in chunks}
    assert 1 not in pages_with_chunks  # empty page skipped


def test_page_text_uses_extract_mode(tmp_pdf):
    """_page_text must forward self.extract_mode to page.get_text()."""
    pages = ["some text"]
    with _patch_fitz(pages) as mock_open:
        fake_doc = mock_open.return_value.__enter__.return_value
        fake_page = _make_fitz_page("some text")
        fake_doc.__getitem__ = lambda self, i: fake_page
        fake_doc.__len__ = lambda self: 1

        PyMuPDFParser(extract_mode="blocks").parse(tmp_pdf, chunk_size=512, overlap=0)

    fake_page.get_text.assert_called_with("blocks")


# ---------------------------------------------------------------------------
# Integration test — requires a real PDF
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_pdf_parse(tmp_path: Path):
    """Generate a minimal PDF with reportlab and parse it."""
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas as rl_canvas

    pdf_path = tmp_path / "test.pdf"
    c = rl_canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Hello from a real PDF! " * 20)
    c.showPage()
    c.save()

    chunks = PyMuPDFParser().parse(pdf_path, chunk_size=100, overlap=20)
    assert len(chunks) >= 1
    assert all(c.source_file == "test.pdf" for c in chunks)
    assert all(len(c.text) <= 100 for c in chunks)
