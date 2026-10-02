"""Shared data-model classes used by both the ingestion pipeline and the MCP server.

Design notes
------------
* ``DocumentEntry`` is the base unit of information — it carries the text payload
  plus optional filter field (`category``) that map directly
  to Qdrant payload filters.

* ``DocumentQuery`` is *generic* over any subclass of ``DocumentEntry`` (bound via
  ``EntryT``).  This means you can extend ``DocumentEntry`` with extra fields and
  pass your subclass straight into ``DocumentQuery`` without touching this file:

      class RichEntry(DocumentEntry):
          author: str

      q = DocumentQuery[RichEntry](
          query=RichEntry(text="transformer architecture", author="Vaswani"),
          max_results=10,
      )
"""
from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base document unit
# ---------------------------------------------------------------------------


class DocumentEntry(BaseModel):
    """A single document unit used for ingestion **or** as a search anchor.

    Attributes:
        text:     The raw text to embed and store (ingestion) or the natural-
                  language query string (search).
        category: Optional label for a logical section or topic within a
                  document — e.g. a chapter name or document section.  Also
                  used as a Qdrant payload filter.
        version:  Optional version string of the document (e.g. ``"v2"`` or
                  ``"2024-Q1"``).  Stored in the Qdrant payload during
                  ingestion and used as a filter during search.
    """

    text: str = Field(..., description="Text content to ingest or search by.")
    category: Optional[str] = Field(
        default=None,
        description="Narrows search to a specific category or section (Qdrant filter).",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version of the document to search (Qdrant filter).",
    )


# ---------------------------------------------------------------------------
# Generic query wrapper
# ---------------------------------------------------------------------------

# Bound to DocumentEntry so only DocumentEntry and its subclasses are accepted.
EntryT = TypeVar("EntryT", bound=DocumentEntry)


class DocumentQuery(BaseModel, Generic[EntryT]):
    """A search request wrapping a :class:`DocumentEntry` (or subclass).

    ``DocumentQuery`` uses *composition* (not inheritance) so the query payload
    and the search parameters stay decoupled.  The generic parameter ``EntryT``
    lets callers substitute any :class:`DocumentEntry` subclass transparently:

    .. code-block:: python

        # Plain usage
        q = DocumentQuery(query=DocumentEntry(text="attention mechanism"), max_results=5)

        # With a richer entry subclass — DocumentQuery needs no changes
        class TaggedEntry(DocumentEntry):
            tags: list[str] = []

        q2 = DocumentQuery[TaggedEntry](
            query=TaggedEntry(text="BERT pre-training", tags=["nlp", "bert"]),
            max_results=3,
        )

    Attributes:
        query:       The :class:`DocumentEntry` (or subclass) that drives the
                     semantic search.
        max_results: Maximum number of results the search API should return.
    """

    query: EntryT
    max_results: int = Field(default=5, ge=1, le=100, description="Number of results to return.")
