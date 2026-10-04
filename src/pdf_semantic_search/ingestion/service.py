"""Ingestion service — orchestrates parse → embed (dense + sparse) → upsert.

The service is the single place that knows about all three sub-systems
(parser, embedders, vector store) and wires them together.  Each sub-system
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
        "category":    str?,  # optional — used for Qdrant filtering
        "version":     str?,  # optional — used for Qdrant filtering
    }

Vector layout per point
-----------------------
Each point contains **two** named vectors:

* ``dense_vector`` — L2-normalised dense embedding (SentenceTransformer).
* ``sparse_vector`` — SPLADE++ sparse embedding (:class:`SparseEmbedding`).

Both are computed in the same batch loop for efficiency.
"""
from __future__ import annotations

import uuid
import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

from pdf_semantic_search.embeddings.sentence_transformer import EmbeddingModel
from pdf_semantic_search.embeddings.sparse_embedder import SparseEmbeddingModel
from pdf_semantic_search.pdf.parser import PDFParserBase, TextChunk
from pdf_semantic_search.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Ties together a parser, a dense embedder, a sparse embedder, and a store.

    Args:
        parser:          Any :class:`~pdf_semantic_search.pdf.parser.PDFParserBase`
                         implementation.
        embedder:        Dense :class:`~pdf_semantic_search.embeddings.sentence_transformer.EmbeddingModel`
                         implementation.
        sparse_embedder: Sparse :class:`~pdf_semantic_search.embeddings.sparse_embedder.SparseEmbeddingModel`
                         implementation (SPLADE-style).
        store:           A :class:`~pdf_semantic_search.vector_store.qdrant_store.QdrantStore` instance.
    """

    def __init__(
        self,
        parser: PDFParserBase,
        embedder: EmbeddingModel,
        sparse_embedder: SparseEmbeddingModel,
        store: QdrantStore,
    ) -> None:
        self.parser = parser
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
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
        category: Optional[str] = None,
        version: Optional[str] = None,
    ) -> int:
        """Parse *pdf_path*, embed chunks (dense + sparse) in batches, and upsert.

        Args:
            pdf_path:   Path to the PDF file to ingest.
            chunk_size: Characters per chunk (forwarded to the parser).
            overlap:    Character overlap between consecutive chunks.
            batch_size: Number of chunks to embed and upsert in one go.
                        Larger values improve throughput; smaller values
                        reduce peak memory usage.
            category:   Optional label stored in every point's payload
                        ``category`` field.  Enables Qdrant filtering.
            version:    Optional version string stored in every point's payload
                        ``version`` field.  Enables Qdrant filtering.

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

        # ── 2. Embed (dense + sparse) + upsert in batches ─────────────────────
        total = 0
        for batch in self._batched(chunks, batch_size):
            texts = [c.text for c in batch]

            # Dense embeddings from SentenceTransformer
            dense_vectors = self.embedder.embed(texts)

            # Sparse embeddings from SPLADE++ (fastembed)
            sparse_vectors = self.sparse_embedder.embed_sparse(texts)

            points = [
                {
                    "id": self._deterministic_id(chunk),
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    "payload": {
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "category": category,
                        "version": version,
                    },
                }
                for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors)
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
