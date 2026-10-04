"""Qdrant vector store — collection management, upsert, and hybrid search.

Payload schema stored alongside every vector
--------------------------------------------
Every point written by :class:`QdrantStore` carries a JSON payload with the
following keys.  All keys are indexed so they can be used as Qdrant filters:

    {
        "text":        str,   # raw chunk text
        "source_file": str,   # originating PDF filename / path
        "page":        int,   # 0-based page index
        "chunk_index": int,   # position within that page's chunks
        "category":    str?,  # optional — maps to DocumentEntry.category
        "version":     str?,  # optional — maps to DocumentEntry.version
    }

Vector layout
-------------
Each point stores **two named vectors**:

* ``"dense"`` (configurable via :attr:`QdrantStore.dense_vector_name`):
  A dense L2-normalised embedding from a SentenceTransformer model.
  Used for semantic / conceptual similarity (cosine distance).

* ``"sparse"`` (configurable via :attr:`QdrantStore.sparse_vector_name`):
  A SPLADE++ sparse vector over the model vocabulary.
  Used for exact / near-exact term matching.

Hybrid search with RRF
-----------------------
:meth:`search` issues two ``Prefetch`` sub-queries (one dense ANN, one sparse)
and fuses them with **Reciprocal Rank Fusion** (``Fusion.RRF``) so that
documents ranking highly in *either* modality surface at the top.

Filter behaviour
----------------
``category`` and ``version`` are forwarded as Qdrant ``must`` conditions so
only points that match *all* supplied filters are returned.  Both Prefetch
legs apply the same filter before RRF merging.

.. note::
    If you have an existing collection that was created with the old
    single (unnamed) dense vector format, you must delete and recreate it
    before this code can ingest or search it.  Run::

        docker compose down -v && docker compose up -d
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from pdf_semantic_search.models import SparseEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single chunk returned from a hybrid (dense + sparse + RRF) search.

    Attributes:
        score:       RRF fusion score (higher = ranked better across both
                     the dense and sparse retrieval legs).
        text:        The raw chunk text stored in Qdrant.
        source_file: Name / path of the source PDF.
        page:        0-based page index within the PDF.
        chunk_index: Position of this chunk within the page.
        category:    Value of the ``category`` payload field (may be ``None``).
        version:     Value of the ``version`` payload field (may be ``None``).
        metadata:    Full Qdrant payload dict for any extra fields.
    """

    score: float
    text: str
    source_file: str
    page: int
    chunk_index: int
    category: Optional[str]
    version: Optional[str]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class QdrantStore:
    """Thin, opinionated wrapper around the Qdrant Python client.

    Responsibilities
    ----------------
    * Create the named collection (dense + sparse named vectors) on first
      connect if it does not already exist.
    * Upsert batches of vector points that carry **both** a dense and a sparse
      vector per chunk.
    * Run hybrid searches (dense ANN + sparse term matching) fused with
      Reciprocal Rank Fusion and return typed :class:`SearchResult` objects.

    The Qdrant client is created lazily — no network call happens until
    :meth:`_connect` is invoked (which is called by :meth:`upsert` and
    :meth:`search` on first use).

    Args:
        host:               Qdrant hostname (e.g. ``"localhost"``).
        port:               Qdrant HTTP port (default ``6333``).
        collection:         Name of the Qdrant collection to use.
        dim:                Dense vector dimension — **must** match the dense
                            embedding model output.
        dense_vector_name:  Name of the dense named-vector field in Qdrant.
                            Defaults to ``"dense"``.
        sparse_vector_name: Name of the sparse named-vector field in Qdrant.
                            Defaults to ``"sparse"``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        dim: int,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
    ) -> None:
        self.host = host
        self.port = port
        self.collection = collection
        self.dim = dim
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self._client = None  # qdrant_client.QdrantClient — set in _connect()

    # ------------------------------------------------------------------
    # Connection & collection bootstrap
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Initialise the Qdrant client and ensure the collection exists.

        Creates the collection with both a dense (cosine) and a sparse named
        vector if it does not yet exist.  Safe to call multiple times —
        subsequent calls are no-ops once the client is set.
        """
        if self._client is not None:
            return  # already connected

        logger.info("Connecting to Qdrant at %s:%d …", self.host, self.port)
        self._client = QdrantClient(host=self.host, port=self.port)

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection not in existing:
            logger.info(
                "Collection %r not found — creating (dim=%d, dense=%r, sparse=%r).",
                self.collection,
                self.dim,
                self.dense_vector_name,
                self.sparse_vector_name,
            )
            self._client.create_collection(
                collection_name=self.collection,
                # Named dense vector (cosine similarity for semantic search)
                vectors_config={
                    self.dense_vector_name: VectorParams(
                        size=self.dim,
                        distance=Distance.COSINE,
                    ),
                },
                # Named sparse vector (dot-product for SPLADE term matching)
                sparse_vectors_config={
                    self.sparse_vector_name: SparseVectorParams(),
                },
            )
            self._ensure_payload_indexes()
        else:
            logger.info("Collection %r already exists — skipping creation.", self.collection)

    def _ensure_payload_indexes(self) -> None:
        """Create keyword indexes on filterable payload fields.

        Called once after the collection is first created.  Qdrant requires
        explicit indexes for efficient payload filtering.
        """
        for field_name in ("category", "version", "source_file"):
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

        - ``"dense_vector"`` — ``list[float]`` of length :attr:`dim`.
        - ``"sparse_vector"`` — :class:`~pdf_semantic_search.models.SparseEmbedding`
          produced by a SPLADE-style sparse embedder.
        - ``"payload"`` — ``dict`` with at minimum ``"text"`` and
          ``"source_file"`` keys.  ``"category"`` and ``"version"`` are
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

        structs = [
            PointStruct(
                id=p.get("id") or str(uuid.uuid4()),
                vector={
                    self.dense_vector_name: p["dense_vector"],
                    self.sparse_vector_name: SparseVector(
                        indices=p["sparse_vector"].indices,
                        values=p["sparse_vector"].values,
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ]

        self._client.upsert(  # type: ignore[union-attr]
            collection_name=self.collection,
            points=structs,
            wait=True,
        )
        logger.debug("Upserted %d point(s) into %r.", len(structs), self.collection)

    # ------------------------------------------------------------------
    # Read — hybrid dense + sparse with RRF fusion
    # ------------------------------------------------------------------

    def search(
        self,
        query_dense: list[float],
        query_sparse: SparseEmbedding,
        top_r: int = 5,
        category: Optional[str] = None,
        version: Optional[str] = None,
    ) -> list[SearchResult]:
        """Return the *top_r* best chunks using hybrid dense + sparse search.

        Issues two ``Prefetch`` sub-queries in parallel:

        1. **Dense ANN** — cosine similarity via the ``"dense"`` named vector.
        2. **Sparse term** — dot-product via the ``"sparse"`` named vector
           (SPLADE++ weights map to vocabulary term relevance).

        Both legs are merged with **Reciprocal Rank Fusion** (RRF), which
        rewards documents that rank highly in *either* (or both) sub-queries.
        This means:

        * Exact / near-exact keyword matches ("hot engine") surface even if
          they are semantically distant in the dense space.
        * Semantically related concepts ("engine warming up") surface even
          without the exact term.

        Args:
            query_dense:  Dense embedding of the search query.
            query_sparse: Sparse SPLADE embedding of the search query.
            top_r:        Maximum number of results to return after RRF fusion.
            category:     If set, restricts *both* sub-queries to points whose
                          payload ``category`` field exactly matches this value.
            version:      If set, restricts *both* sub-queries to points whose
                          payload ``version`` field exactly matches this value.

        Returns:
            List of :class:`SearchResult` objects ordered by descending RRF
            score (best match first).
        """
        self._connect()

        query_filter = self._build_filter(category=category, version=version)

        # Prefetch more candidates than `top_r` so RRF has enough material
        # to re-rank.  A factor of 3× with a minimum of 20 is a safe default.
        prefetch_limit = max(top_r * 3, 20)

        logger.debug(
            "Hybrid search in %r — top_r=%d, prefetch=%d, category=%r, version=%r",
            self.collection,
            top_r,
            prefetch_limit,
            category,
            version,
        )

        response = self._client.query_points(  # type: ignore[union-attr]
            collection_name=self.collection,
            prefetch=[
                # Leg 1: dense semantic ANN
                Prefetch(
                    query=query_dense,
                    using=self.dense_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                # Leg 2: sparse term matching (SPLADE)
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse.indices,
                        values=query_sparse.values,
                    ),
                    using=self.sparse_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_r,
            with_payload=True,
        )

        return [self._hit_to_result(hit) for hit in response.points]

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
        offset = None

        while True:
            records, offset = self._client.scroll(  # type: ignore[union-attr]
                collection_name=self.collection,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=["category"],
                with_vectors=False,
            )

            for record in records:
                cat = (record.payload or {}).get("category")
                if cat:
                    categories.add(cat)

            if offset is None:
                break

        return sorted(categories)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(
        category: Optional[str],
        version: Optional[str] = None,
    ):
        """Build a Qdrant ``Filter`` from optional keyword constraints.

        Returns ``None`` if no fields are set (no filtering applied).
        The same filter is applied to *both* Prefetch legs so results stay
        consistent across dense and sparse retrieval.
        """
        must_filters: list[Condition] = []

        if category is not None:
            must_filters.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if version is not None:
            must_filters.append(
                FieldCondition(key="version", match=MatchValue(value=version))
            )

        return Filter(must=must_filters) if must_filters else None

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
            category=payload.get("category"),
            version=payload.get("version"),
            metadata=payload,
        )
