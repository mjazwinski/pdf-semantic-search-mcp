"""Qdrant vector store — collection management, upsert, and filtered search.

Payload schema stored alongside every vector
--------------------------------------------
Every point written by :class:`QdrantStore` carries a JSON payload with the
following keys.  All keys are indexed so they can be used as Qdrant filters:

    {
        "text":        str,   # raw chunk text
        "source_file": str,   # originating PDF filename / path
        "page":        int,   # 0-based page index
        "chunk_index": int,   # position within that page's chunks
        "context":     str?,  # optional — maps to DocumentEntry.context
        "category":    str?,  # optional — maps to DocumentEntry.category
    }

Filter behaviour in search()
-----------------------------
``context`` and ``category`` are forwarded as Qdrant ``must`` conditions so
only points that match *all* supplied filters are returned.  Omitting a filter
field (``None``) means "no restriction on this field".
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single chunk returned from a semantic search.

    Attributes:
        score:       Cosine similarity score in [0, 1] (higher = more relevant).
        text:        The raw chunk text stored in Qdrant.
        source_file: Name / path of the source PDF.
        page:        0-based page index within the PDF.
        chunk_index: Position of this chunk within the page.
        context:     Value of the ``context`` payload field (may be ``None``).
        category:    Value of the ``category`` payload field (may be ``None``).
        metadata:    Full Qdrant payload dict for any extra fields.
    """

    score: float
    text: str
    source_file: str
    page: int
    chunk_index: int
    context: Optional[str]
    category: Optional[str]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class QdrantStore:
    """Thin, opinionated wrapper around the Qdrant Python client.

    Responsibilities:
    - Create the named collection (with cosine distance) on first connect if
      it does not already exist.
    - Upsert batches of vector points with structured payloads.
    - Run filtered semantic searches and return typed :class:`SearchResult` objects.

    The client is created lazily — no network call happens until
    :meth:`_connect` is invoked (which is called by :meth:`upsert` and
    :meth:`search` on first use).

    Args:
        host:       Qdrant hostname (e.g. ``"localhost"``).
        port:       Qdrant HTTP port (default ``6333``).
        collection: Name of the Qdrant collection to use.
        dim:        Vector dimension — **must** match the embedding model output.
    """

    def __init__(self, host: str, port: int, collection: str, dim: int) -> None:
        self.host = host
        self.port = port
        self.collection = collection
        self.dim = dim
        self._client = None  # qdrant_client.QdrantClient — set in _connect()

    # ------------------------------------------------------------------
    # Connection & collection bootstrap
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Initialise the Qdrant client and ensure the collection exists.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._client is not None:
            return  # already connected

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        logger.info("Connecting to Qdrant at %s:%d …", self.host, self.port)
        self._client = QdrantClient(host=self.host, port=self.port)

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection not in existing:
            logger.info(
                "Collection %r not found — creating with dim=%d, distance=Cosine.",
                self.collection,
                self.dim,
            )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            # Index payload fields used as filters so queries stay fast
            self._ensure_payload_indexes()
        else:
            logger.info("Collection %r already exists — skipping creation.", self.collection)

    def _ensure_payload_indexes(self) -> None:
        """Create keyword indexes on filterable payload fields.

        Called once after the collection is first created.  Qdrant requires
        explicit indexes for efficient payload filtering.
        """
        from qdrant_client.models import PayloadSchemaType

        for field_name in ("context", "category", "source_file"):
            self._client.create_payload_index(  # type: ignore[union-attr]
                collection_name=self.collection,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.debug("Created keyword index on payload field %r.", field_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: list[dict]) -> None:
        """Insert or update vector points in the collection.

        Each dict in *points* must contain:

        - ``"vector"`` — ``list[float]`` of length :attr:`dim`.
        - ``"payload"`` — ``dict`` with at minimum ``"text"`` and
          ``"source_file"`` keys.  ``"context"`` and ``"category"`` are
          optional but enable filtering.
        - ``"id"`` — *optional* ``int`` or ``str`` UUID.  A random UUID is
          generated if omitted.

        Args:
            points: List of point dicts as described above.

        Raises:
            ValueError: If *points* is empty.
        """
        if not points:
            raise ValueError("points must contain at least one item.")

        self._connect()

        from qdrant_client.models import PointStruct

        structs = [
            PointStruct(
                id=p.get("id") or str(uuid.uuid4()),
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points
        ]

        self._client.upsert(  # type: ignore[union-attr]
            collection_name=self.collection,
            points=structs,
            wait=True,  # block until the operation is acknowledged
        )
        logger.debug("Upserted %d point(s) into %r.", len(structs), self.collection)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        context: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most similar chunks to *query_vector*.

        Args:
            query_vector: Embedding of the search query (same dim as stored vectors).
            top_k:        Maximum number of results to return.
            context:      If set, restricts results to points whose payload
                          ``context`` field exactly matches this value.
            category:     If set, restricts results to points whose payload
                          ``category`` field exactly matches this value.

        Returns:
            List of :class:`SearchResult` objects ordered by descending score.
        """
        self._connect()

        query_filter = self._build_filter(context=context, category=category)

        logger.debug(
            "Searching %r — top_k=%d, context=%r, category=%r",
            self.collection,
            top_k,
            context,
            category,
        )

        hits = self._client.search(  # type: ignore[union-attr]
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [self._hit_to_result(h) for h in hits]

    def list_categories(self) -> list[str]:
        """Return all distinct ``category`` values present in the collection.

        Uses Qdrant's scroll API with a payload selector to avoid loading
        vectors.  Results are deduplicated and sorted alphabetically.

        Returns:
            Sorted list of unique category strings.  Empty strings and
            ``None`` values are excluded.
        """
        self._connect()

        categories: set[str] = set()
        offset = None  # scroll cursor

        while True:
            records, offset = self._client.scroll(  # type: ignore[union-attr]
                collection_name=self.collection,
                scroll_filter=None,
                limit=256,          # page size for the scroll
                offset=offset,
                with_payload=["category"],
                with_vectors=False,
            )

            for record in records:
                cat = (record.payload or {}).get("category")
                if cat:
                    categories.add(cat)

            if offset is None:
                break  # no more pages

        return sorted(categories)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(
        context: Optional[str],
        category: Optional[str],
    ):
        """Build a Qdrant ``Filter`` from optional keyword constraints.

        Returns ``None`` if neither field is set (no filtering applied).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must: list[FieldCondition] = []

        if context is not None:
            must.append(
                FieldCondition(key="context", match=MatchValue(value=context))
            )
        if category is not None:
            must.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )

        return Filter(must=must) if must else None

    @staticmethod
    def _hit_to_result(hit) -> SearchResult:
        """Convert a raw Qdrant ``ScoredPoint`` to a :class:`SearchResult`."""
        payload = hit.payload or {}
        return SearchResult(
            score=hit.score,
            text=payload.get("text", ""),
            source_file=payload.get("source_file", ""),
            page=payload.get("page", 0),
            chunk_index=payload.get("chunk_index", 0),
            context=payload.get("context"),
            category=payload.get("category"),
            metadata=payload,
        )
