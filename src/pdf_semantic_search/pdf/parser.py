"""PDF parser interface and default PyMuPDF implementation.

Extension guide
---------------
Provide a custom extraction strategy (OCR, table-aware, layout-preserving, …)
by subclassing :class:`PDFParserBase` and implementing :meth:`parse`.  Pass
your subclass to :class:`~pdf_semantic_search.ingestion.service.IngestionService`
at construction time — no other changes required.

    class MyOCRParser(PDFParserBase):
        def parse(self, pdf_path, chunk_size=512, overlap=64):
            ...  # custom logic
            return [TextChunk(...), ...]

    service = IngestionService(parser=MyOCRParser(), embedder=..., store=...)
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from pathlib import Path

from pdf_semantic_search.pdf.chunker import chunk_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A piece of text extracted from a PDF with full provenance metadata.

    Attributes:
        text:        The chunk's raw text content.
        source_file: Filename (stem + suffix) of the originating PDF.
        page:        0-based page index within the PDF.
        chunk_index: Position of this chunk among all chunks on *page*.
    """

    text: str
    source_file: str
    page: int
    chunk_index: int


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class PDFParserBase(abc.ABC):
    """Interface every PDF parser implementation must satisfy.

    A parser is responsible for:

    - Opening and reading the PDF at *pdf_path*.
    - Extracting plain text (however it sees fit).
    - Splitting that text into :class:`TextChunk` objects using the supplied
      *chunk_size* / *overlap* parameters.

    The :class:`IngestionService` calls :meth:`parse` and consumes the result;
    it is agnostic to the implementation details.
    """

    @abc.abstractmethod
    def parse(
        self,
        pdf_path: Path,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[TextChunk]:
        """Parse *pdf_path* and return an ordered list of :class:`TextChunk` objects.

        Args:
            pdf_path:   Absolute or relative path to a readable PDF file.
            chunk_size: Target maximum characters per chunk.
            overlap:    Overlap in characters between consecutive chunks.

        Returns:
            Flat, ordered list of :class:`TextChunk` objects across all pages.
            Empty pages produce no chunks.
        """


# ---------------------------------------------------------------------------
# Default implementation — PyMuPDF
# ---------------------------------------------------------------------------


class PyMuPDFParser(PDFParserBase):
    """Default parser using PyMuPDF (``fitz``) for text extraction.

    Extraction strategy
    -------------------
    For each page the parser calls ``page.get_text("text")`` which returns
    plain, UTF-8 text in reading order (left-to-right, top-to-bottom).
    The raw page text is then passed to :func:`~pdf_semantic_search.pdf.chunker.chunk_text`
    for sliding-window chunking.

    This strategy works well for text-based PDFs.  For scanned documents
    (image-only PDFs) use an OCR-aware subclass instead.

    Args:
        extract_mode: PyMuPDF text extraction mode passed to ``get_text()``.
                      ``"text"`` (default) returns plain text.  ``"blocks"``
                      or ``"words"`` can be used by subclasses that override
                      :meth:`_page_text`.
    """

    def __init__(self, extract_mode: str = "text") -> None:
        self.extract_mode = extract_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        pdf_path: Path,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[TextChunk]:
        """Open *pdf_path* with PyMuPDF and chunk every page's text.

        Args:
            pdf_path:   Path to the PDF file.
            chunk_size: Target chunk length in characters (passed to chunker).
            overlap:    Character overlap between consecutive chunks.

        Returns:
            Ordered list of :class:`TextChunk` covering all non-empty pages.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
            RuntimeError:      If PyMuPDF cannot open the file.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        import fitz  # PyMuPDF — imported here to keep module import fast

        result: list[TextChunk] = []
        source_file = pdf_path.name  # store just the filename, not full path

        logger.info("Parsing %r …", str(pdf_path))

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise RuntimeError(f"PyMuPDF could not open {pdf_path}: {exc}") from exc

        with doc:
            total_pages = len(doc)
            logger.debug("  %d page(s) found.", total_pages)

            for page_index in range(total_pages):
                page = doc[page_index]
                raw_text = self._page_text(page)

                if not raw_text.strip():
                    logger.debug("  Page %d is empty — skipping.", page_index)
                    continue

                page_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                logger.debug(
                    "  Page %d → %d chunk(s).", page_index, len(page_chunks)
                )

                for chunk_index, text in enumerate(page_chunks):
                    result.append(
                        TextChunk(
                            text=text,
                            source_file=source_file,
                            page=page_index,
                            chunk_index=chunk_index,
                        )
                    )

        logger.info("Parsed %r — %d chunk(s) total.", source_file, len(result))
        return result

    # ------------------------------------------------------------------
    # Override point for subclasses
    # ------------------------------------------------------------------

    def _page_text(self, page) -> str:
        """Extract plain text from a single PyMuPDF page object.

        Override this in a subclass to apply custom extraction logic
        (e.g. filtering out headers/footers by bounding-box coordinates).

        Args:
            page: A ``fitz.Page`` object.

        Returns:
            Raw text string for the page.
        """
        return page.get_text(self.extract_mode)
