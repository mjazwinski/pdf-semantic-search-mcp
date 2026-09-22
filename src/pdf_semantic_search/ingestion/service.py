"""Ingestion service — orchestrates parse → embed → upsert.

The service is the single place that knows about all three sub-systems
(parser, embedder, vector store) and wires them together.  Each sub-system
is injected at construction time, making the class fully testable without
a live Qdrant instance or a GPU.

Payload schema written to Qdrant
---------------------------------
Each point's payload mirrors :class:`~pdf_semantic_search.pdf.parser.TextChunk`
plus the two optional filter fields from
:class:`~pdf_semantic_search.models.DocumentEntry`:

    {
        "text":        str,   # chunk text
        "source_file": str,   # PDF filename
        "page":        int,   # 0-based page index
        "chunk_index": int,   # position within page
        "context":     str?,  # optional — used for Qdrant filtering
        "category":    str?,  # optional — used for Qdrant filtering
    }
"""
from __future__ import annotations

import uuid
import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

from pdf_semantic_search.embeddings.sentence_transformer import EmbeddingModel
from pdf_semantic_search.pdf.parser import PDFParserBase, TextChunk
from pdf_semantic_search.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Ties together a parser, an embedding model, and a vector store.

    Args:
        parser:   Any :class:`~pdf_semantic_search.pdf.parser.PDFParserBase` implementation.
        embedder: Any :class:`~pdf_semantic_search.embeddings.sentence_transformer.EmbeddingModel`
                  implementation.
        store:    A :class:`~pdf_semantic_search.vector_store.qdrant_store.QdrantStore` instance.
    """

    def __init__(
        self,
        parser: PDFParserBase,
        embedder: EmbeddingModel,
        store: QdrantStore,
    ) -> None:
        self.parser = parser
        self.embedder = embedder
        self.store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_path: Path,
        chunk_size: int = 512,
        overlap: int = 64,
        batch_size: int = 64,
        context: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        """Parse *pdf_path*, embed chunks in batches, and upsert to Qdrant.

        Args:
            pdf_path:   Path to the PDF file to ingest.
            chunk_size: Characters per chunk (forwarded to the parser).
            overlap:    Character overlap between consecutive chunks.
            batch_size: Number of chunks to embed and upsert in one go.
                        Larger values improve throughput; smaller values
                        reduce peak memory usage.
            context:    Optional label stored in every point's payload
                        ``context`` field (e.g. the document name / title).
                        Enables Qdrant filtering on this field at search time.
            category:   Optional label stored in every point's payload
                        ``category`` field (e.g. a document section or topic).
                        Enables Qdrant filtering on this field at search time.

        Returns:
            Total number of chunks ingested (upserted to Qdrant).

        Raises:
            FileNotFoundError: Propagated from the parser if the file is missing.
        """
        pdf_path = Path(pdf_path)
        logger.info(
            "Starting ingestion: file=%r, chunk_size=%d, overlap=%d, batch_size=%d",
            str(pdf_path),
            chunk_size,
            overlap,
            batch_size,
        )

        # ── 1. Parse ─────────────────────────────────────────────────────────
        chunks: list[TextChunk] = self.parser.parse(
            pdf_path, chunk_size=chunk_size, overlap=overlap
        )
        logger.info("Parser produced %d chunk(s).", len(chunks))

        if not chunks:
            logger.warning("No chunks produced for %r — nothing ingested.", str(pdf_path))
            return 0

        # ── 2. Embed + upsert in batches ─────────────────────────────────────
        total = 0
        for batch in self._batched(chunks, batch_size): # the generartor should be build in in the parser if it makes sense at all. here the document is already in memory
            texts = [c.text for c in batch]
            vectors = self.embedder.embed(texts)

            points = [
                {
                    "id": self._deterministic_id(chunk),
                    "vector": vector,
                    "payload": {
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "context": context,
                        "category": category,
                    },
                }
                for chunk, vector in zip(batch, vectors)
            ]

            self.store.upsert(points)
            total += len(points)
            logger.debug("Upserted batch of %d — running total: %d", len(points), total)

        logger.info("Ingestion complete — %d chunk(s) stored.", total)
        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batched(items: list, size: int) -> Iterator[list]:
        """Yield successive fixed-size slices of *items*."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    @staticmethod
    def _deterministic_id(chunk: TextChunk) -> str:
        """Generate a stable, Qdrant-compatible UUID from chunk provenance.

        Using a deterministic ID means re-ingesting the same PDF with the
        same parameters will *upsert* (overwrite) existing points rather
        than creating duplicates.

        Qdrant requires IDs to be a non-negative integer or a valid UUID.
        We take the first 16 bytes of SHA-1(``source_file:page:chunk_index``)
        and convert them to a UUID — deterministic and collision-resistant for
        any realistic corpus size.
        """
        key = f"{chunk.source_file}:{chunk.page}:{chunk.chunk_index}"
        digest_bytes = hashlib.sha1(key.encode()).digest()[:16]
        return str(uuid.UUID(bytes=digest_bytes))
