"""Tests for DocumentEntry and DocumentQuery data models."""
import pytest
from pydantic import ValidationError

from pdf_semantic_search.models import DocumentEntry, DocumentQuery


# ---------------------------------------------------------------------------
# DocumentEntry
# ---------------------------------------------------------------------------


def test_entry_requires_text():
    with pytest.raises(ValidationError):
        DocumentEntry()  # type: ignore[call-arg]


def test_entry_optional_fields_default_to_none():
    entry = DocumentEntry(text="hello")
    assert entry.context is None
    assert entry.category is None


def test_entry_with_all_fields():
    entry = DocumentEntry(text="hello", context="doc.pdf", category="intro")
    assert entry.text == "hello"
    assert entry.context == "doc.pdf"
    assert entry.category == "intro"


# ---------------------------------------------------------------------------
# DocumentQuery
# ---------------------------------------------------------------------------


def test_query_default_max_results():
    q = DocumentQuery(query=DocumentEntry(text="search term"))
    assert q.max_results == 5


def test_query_custom_max_results():
    q = DocumentQuery(query=DocumentEntry(text="search term"), max_results=10)
    assert q.max_results == 10


def test_query_max_results_bounds():
    with pytest.raises(ValidationError):
        DocumentQuery(query=DocumentEntry(text="x"), max_results=0)
    with pytest.raises(ValidationError):
        DocumentQuery(query=DocumentEntry(text="x"), max_results=101)


def test_query_works_with_entry_subclass():
    """DocumentQuery must accept any DocumentEntry subclass without modification."""

    class RichEntry(DocumentEntry):
        tags: list[str] = []

    rich = RichEntry(text="BERT", context="paper.pdf", tags=["nlp"])
    q: DocumentQuery[RichEntry] = DocumentQuery(query=rich, max_results=3)

    assert q.query.tags == ["nlp"]
    assert q.max_results == 3
