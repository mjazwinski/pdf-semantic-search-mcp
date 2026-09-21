"""End-to-end pipeline tests.

These tests wire the real classes together (no stubs) but mock the three
external I/O boundaries:

    fitz.open      — so no real PDF is needed
    SentenceTransformer.encode  — so no model download is needed
    QdrantClient   — so no Docker container is needed

This gives us confidence that the layers integrate correctly without the
cost of external services.

True integration tests (real Qdrant + real model) are at the bottom and
require ``pytest -m integration`` to run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import make_chunks, make_mock_embedder, make_search_result


# ---------------------------------------------------------------------------
# Helpers — shared fake fitz document
# ---------------------------------------------------------------------------


def _fake_fitz_page(text: str) -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _fake_fitz_doc(pages: list[str]) -> MagicMock:
    doc = MagicMock()
    doc.__len__ = lambda self: len(pages)
    doc.__getitem__ = lambda self, i: _fake_fitz_page(pages[i])
    doc.__enter__ = lambda self: self
    doc.__exit__ = MagicMock(return_value=False)
    return doc


# ---------------------------------------------------------------------------
# Layer 1 + 2: Parser → IngestionService
# Verifies that chunks produced by the parser are correctly turned into
# Qdrant points with the right payload and deterministic IDs.
# ---------------------------------------------------------------------------


class TestParserToIngestion:
    """Wire PyMuPDFParser → IngestionService with a mock Qdrant store."""

    def _run(self, pages: list[str], chunk_size: int = 512, **ingest_kwargs):
        from pdf_semantic_search.ingestion.service import IngestionService
        from pdf_semantic_search.pdf.parser import PyMuPDFParser
        from pdf_semantic_search.vector_store.qdrant_store import QdrantStore

        fake_doc = _fake_fitz_doc(pages)
        embedder = make_mock_embedder(dim=4)
        mock_store = MagicMock(spec=QdrantStore)

        # Provide a real (temp) path so FileNotFoundError isn't raised
        pdf_path = Path("/fake/doc.pdf")

        with patch("fitz.open", return_value=fake_doc):
            with patch.object(Path, "exists", return_value=True):
                service = IngestionService(
                    parser=PyMuPDFParser(),
                    embedder=embedder,
                    store=mock_store,
                )
                total = service.ingest(
                    pdf_path, chunk_size=chunk_size, **ingest_kwargs
                )

        return total, mock_store, embedder

    def test_single_page_short_text_produces_one_chunk(self):
        total, store, _ = self._run(["Hello, this is a short page."])
        assert total == 1
        store.upsert.assert_called_once()
        points = store.upsert.call_args.args[0]
        assert len(points) == 1

    def test_payload_text_matches_page_content(self):
        total, store, _ = self._run(["The transformer architecture revolutionised NLP."])
        payload = store.upsert.call_args.args[0][0]["payload"]
        assert "transformer" in payload["text"]

    def test_payload_source_file_is_filename(self):
        _, store, _ = self._run(["some text"])
        payload = store.upsert.call_args.args[0][0]["payload"]
        assert payload["source_file"] == "doc.pdf"

    def test_payload_page_index_correct(self):
        _, store, _ = self._run(["page zero", "page one"])
        all_points = [
            p
            for call in store.upsert.call_args_list
            for p in call.args[0]
        ]
        pages = {p["payload"]["page"] for p in all_points}
        assert pages == {0, 1}

    def test_context_and_category_stored_in_payload(self):
        _, store, _ = self._run(
            ["content"], context="annual-report", category="financials"
        )
        payload = store.upsert.call_args.args[0][0]["payload"]
        assert payload["context"] == "annual-report"
        assert payload["category"] == "financials"

    def test_deterministic_ids_on_reingest(self):
        fake_doc = _fake_fitz_doc(["some text content here"])
        embedder = make_mock_embedder(dim=4)
        mock_store = MagicMock()

        def run():
            from pdf_semantic_search.ingestion.service import IngestionService
            from pdf_semantic_search.pdf.parser import PyMuPDFParser
            svc = IngestionService(PyMuPDFParser(), embedder, mock_store)
            with patch("fitz.open", return_value=fake_doc):
                with patch.object(Path, "exists", return_value=True):
                    svc.ingest(Path("/fake/doc.pdf"))
            return [p["id"] for p in mock_store.upsert.call_args.args[0]]

        ids_first = run()
        mock_store.reset_mock()
        ids_second = run()
        assert ids_first == ids_second

    def test_empty_pdf_produces_no_upsert(self):
        total, store, embedder = self._run(["   ", "\n\n"])
        assert total == 0
        store.upsert.assert_not_called()
        embedder.embed.assert_not_called()

    def test_multi_page_embed_called_with_all_texts(self):
        pages = ["alpha beta gamma", "delta epsilon zeta"]
        _, store, embedder = self._run(pages, chunk_size=512)
        all_texts = [
            text
            for call in embedder.embed.call_args_list
            for text in call.args[0]
        ]
        assert len(all_texts) == 2

    def test_vector_dimension_stored_in_point(self):
        _, store, _ = self._run(["embedding dimension test"], chunk_size=512)
        point = store.upsert.call_args.args[0][0]
        assert len(point["vector"]) == 4  # dim=4 in make_mock_embedder


# ---------------------------------------------------------------------------
# Layer 2 + 3: IngestionService → MCP search_docs
# Verifies that a query flowing through the MCP handler produces the correct
# JSON response, and that filter fields propagate end-to-end.
# ---------------------------------------------------------------------------


class TestIngestionToMCPSearch:
    """Wire mock embedder + mock store → MCP search_docs handler."""

    @pytest.fixture(autouse=True)
    def reset_deps(self):
        from pdf_semantic_search.mcp_server.server import _reset_deps
        _reset_deps()
        yield
        _reset_deps()

    def _make_deps(self, results=None, categories=None, dim=4):
        from pdf_semantic_search.mcp_server.server import _Deps
        embedder = make_mock_embedder(dim=dim)
        store = MagicMock()
        store.search.return_value = results or []
        store.list_categories.return_value = categories or []
        return _Deps(embedder=embedder, store=store)

    @pytest.mark.asyncio
    async def test_search_result_fields_present_in_json(self):
        from pdf_semantic_search.mcp_server.server import _handle_search_docs

        result = make_search_result(score=0.95, text="NLP chunk", page=1)
        deps = self._make_deps(results=[result])

        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            response = await _handle_search_docs({"query": {"text": "NLP"}})

        item = json.loads(response[0].text)[0]
        assert item["score"] == 0.95
        assert item["text"] == "NLP chunk"
        assert item["page"] == 1
        assert "source_file" in item
        assert "context" in item
        assert "category" in item

    @pytest.mark.asyncio
    async def test_context_filter_propagates_to_store(self):
        from pdf_semantic_search.mcp_server.server import _handle_search_docs

        deps = self._make_deps()
        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            await _handle_search_docs({
                "query": {"text": "test", "context": "q4-report.pdf"},
                "max_results": 3,
            })

        kwargs = deps.store.search.call_args.kwargs
        assert kwargs["context"] == "q4-report.pdf"
        assert kwargs["top_k"] == 3

    @pytest.mark.asyncio
    async def test_category_filter_propagates_to_store(self):
        from pdf_semantic_search.mcp_server.server import _handle_search_docs

        deps = self._make_deps()
        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            await _handle_search_docs({
                "query": {"text": "test", "category": "conclusions"},
            })

        assert deps.store.search.call_args.kwargs["category"] == "conclusions"

    @pytest.mark.asyncio
    async def test_list_categories_returns_json_array(self):
        from pdf_semantic_search.mcp_server.server import _handle_list_categories

        deps = self._make_deps(categories=["conclusions", "intro", "methods"])
        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            response = await _handle_list_categories()

        cats = json.loads(response[0].text)
        assert cats == ["conclusions", "intro", "methods"]

    @pytest.mark.asyncio
    async def test_no_results_returns_empty_json_array(self):
        from pdf_semantic_search.mcp_server.server import _handle_search_docs

        deps = self._make_deps(results=[])
        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            response = await _handle_search_docs({"query": {"text": "obscure topic"}})

        assert json.loads(response[0].text) == []


# ---------------------------------------------------------------------------
# Full stack: Parser → Ingestion → Qdrant payload → MCP search
# Verifies that the payload shape written during ingestion is consistent with
# what the MCP handler reads back from search results.
# ---------------------------------------------------------------------------


class TestFullStackPayloadConsistency:
    """Check that the payload written by IngestionService matches what
    the MCP handler exposes in search results — no field name mismatches."""

    @pytest.fixture(autouse=True)
    def reset_deps(self):
        from pdf_semantic_search.mcp_server.server import _reset_deps
        _reset_deps()
        yield
        _reset_deps()

    @pytest.mark.asyncio
    async def test_ingested_payload_fields_match_search_result_fields(self):
        """Ingest one chunk, capture the payload written, then simulate a
        search returning that payload and verify the MCP JSON contains it."""
        from pdf_semantic_search.ingestion.service import IngestionService
        from pdf_semantic_search.mcp_server.server import _Deps, _handle_search_docs
        from pdf_semantic_search.pdf.parser import PyMuPDFParser
        from pdf_semantic_search.vector_store.qdrant_store import SearchResult

        # ── Ingestion side ────────────────────────────────────────────────────
        fake_doc = _fake_fitz_doc(["A test sentence for the pipeline."])
        embedder = make_mock_embedder(dim=4)
        captured_points: list[dict] = []

        mock_store = MagicMock()
        mock_store.upsert.side_effect = lambda pts: captured_points.extend(pts)

        with patch("fitz.open", return_value=fake_doc):
            with patch.object(Path, "exists", return_value=True):
                svc = IngestionService(PyMuPDFParser(), embedder, mock_store)
                svc.ingest(
                    Path("/fake/test.pdf"),
                    context="test.pdf",
                    category="testing",
                )

        assert captured_points, "No points were ingested"
        ingested_payload = captured_points[0]["payload"]
        ingested_vector  = captured_points[0]["vector"]

        # ── Search side — simulate Qdrant returning the ingested point ────────
        fake_result = SearchResult(
            score=0.99,
            text=ingested_payload["text"],
            source_file=ingested_payload["source_file"],
            page=ingested_payload["page"],
            chunk_index=ingested_payload["chunk_index"],
            context=ingested_payload["context"],
            category=ingested_payload["category"],
            metadata=ingested_payload,
        )

        search_embedder = make_mock_embedder(dim=4)
        search_store = MagicMock()
        search_store.search.return_value = [fake_result]

        deps = _Deps(embedder=search_embedder, store=search_store)

        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            response = await _handle_search_docs({"query": {"text": "test sentence"}})

        item = json.loads(response[0].text)[0]

        # Verify round-trip consistency
        assert item["text"]        == ingested_payload["text"]
        assert item["source_file"] == ingested_payload["source_file"]
        assert item["page"]        == ingested_payload["page"]
        assert item["chunk_index"] == ingested_payload["chunk_index"]
        assert item["context"]     == ingested_payload["context"]
        assert item["category"]    == ingested_payload["category"]
        assert item["score"]       == 0.99


# ---------------------------------------------------------------------------
# True integration tests — require live Qdrant + model download
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrueIntegration:
    """Full pipeline against a real Qdrant container and real model.

    Start Qdrant before running:
        docker compose up -d

    Run with:
        pytest -m integration
    """

    COLLECTION = "pytest_e2e"
    DIM = 384
    MODEL = "all-MiniLM-L6-v2"

    @pytest.fixture(autouse=True)
    def cleanup_collection(self):
        """Drop the test collection before and after each test."""
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.recreate_collection(  # type: ignore
            collection_name=self.COLLECTION,
            vectors_config={"size": self.DIM, "distance": "Cosine"},
        )
        yield
        try:
            client.delete_collection(self.COLLECTION)
        except Exception:
            pass

    def test_ingest_and_search_roundtrip(self, tmp_path):
        import fitz

        from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder
        from pdf_semantic_search.ingestion.service import IngestionService
        from pdf_semantic_search.pdf.parser import PyMuPDFParser
        from pdf_semantic_search.vector_store.qdrant_store import QdrantStore

        # Create a minimal real PDF
        pdf_path = tmp_path / "integration.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Transformer models use self-attention mechanisms. " * 10)
        doc.save(str(pdf_path))
        doc.close()

        embedder = SentenceTransformerEmbedder(model_name=self.MODEL)
        store = QdrantStore(
            host="localhost", port=6333,
            collection=self.COLLECTION, dim=self.DIM,
        )
        store._connect()

        svc = IngestionService(parser=PyMuPDFParser(), embedder=embedder, store=store)
        total = svc.ingest(pdf_path, chunk_size=200, overlap=20,
                           context="integration.pdf", category="test")
        assert total > 0

        query_vec = embedder.embed(["self-attention"])[0]
        results = store.search(query_vec, top_k=3, context="integration.pdf")

        assert len(results) > 0
        assert results[0].score > 0.5
        assert "attention" in results[0].text.lower()

    @pytest.mark.asyncio
    async def test_mcp_search_against_live_qdrant(self, tmp_path):
        import fitz

        from pdf_semantic_search.embeddings.sentence_transformer import SentenceTransformerEmbedder
        from pdf_semantic_search.ingestion.service import IngestionService
        from pdf_semantic_search.mcp_server.server import _Deps, _handle_search_docs, _reset_deps
        from pdf_semantic_search.pdf.parser import PyMuPDFParser
        from pdf_semantic_search.vector_store.qdrant_store import QdrantStore

        _reset_deps()

        pdf_path = tmp_path / "mcp_test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Neural networks learn representations from data. " * 8)
        doc.save(str(pdf_path))
        doc.close()

        embedder = SentenceTransformerEmbedder(model_name=self.MODEL)
        store = QdrantStore("localhost", 6333, self.COLLECTION, self.DIM)
        store._connect()

        svc = IngestionService(PyMuPDFParser(), embedder, store)
        svc.ingest(pdf_path, chunk_size=200, overlap=20,
                   context="mcp_test.pdf", category="ml")

        deps = _Deps(embedder=embedder, store=store)
        with patch("pdf_semantic_search.mcp_server.server._get_deps", return_value=deps):
            response = await _handle_search_docs({
                "query": {"text": "deep learning representations", "category": "ml"},
                "max_results": 3,
            })

        results = json.loads(response[0].text)
        assert len(results) > 0
        assert results[0]["score"] > 0.5
        assert results[0]["category"] == "ml"

        _reset_deps()
