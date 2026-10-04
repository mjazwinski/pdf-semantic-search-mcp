"""Tests for IngestionService and the CLI.

Unit tests mock all four dependencies (parser, dense embedder, sparse embedder,
store) so the suite runs without a live Qdrant instance, a GPU, or a real PDF.

Integration tests are gated behind ``@pytest.mark.integration``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pdf_semantic_search.ingestion.cli import app
from pdf_semantic_search.ingestion.service import IngestionService
from pdf_semantic_search.models import SparseEmbedding
from pdf_semantic_search.pdf.parser import TextChunk

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_chunks(n: int, page: int = 0) -> list[TextChunk]:
    return [
        TextChunk(text=f"chunk text {i}", source_file="doc.pdf", page=page, chunk_index=i)
        for i in range(n)
    ]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return a deterministic dense vector for each text (length-based)."""
    return [[float(len(t))] * 4 for t in texts]


def _fake_sparse_embed(texts: list[str]) -> list[SparseEmbedding]:
    """Return a deterministic single-entry sparse vector for each text."""
    return [SparseEmbedding(indices=[len(t) % 100], values=[1.0]) for t in texts]


def _make_service(
    chunks: list[TextChunk],
) -> tuple[IngestionService, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Wire up a service with mocked parser, dense embedder, sparse embedder, store."""
    parser = MagicMock()
    parser.parse.return_value = chunks

    embedder = MagicMock()
    embedder.embed.side_effect = _fake_embed

    sparse_embedder = MagicMock()
    sparse_embedder.embed_sparse.side_effect = _fake_sparse_embed

    store = MagicMock()

    service = IngestionService(
        parser=parser,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        store=store,
    )
    return service, parser, embedder, sparse_embedder, store


# ---------------------------------------------------------------------------
# IngestionService.ingest — core behaviour
# ---------------------------------------------------------------------------


def test_ingest_returns_chunk_count():
    service, *_ = _make_service(_make_chunks(10))
    result = service.ingest(Path("doc.pdf"))
    assert result == 10


def test_ingest_calls_parser_with_correct_args():
    service, parser, *_ = _make_service(_make_chunks(3))
    service.ingest(Path("doc.pdf"), chunk_size=256, overlap=32)
    parser.parse.assert_called_once_with(Path("doc.pdf"), chunk_size=256, overlap=32)


def test_ingest_calls_dense_embed_with_chunk_texts():
    chunks = _make_chunks(3)
    service, _, embedder, *_ = _make_service(chunks)
    service.ingest(Path("doc.pdf"), batch_size=10)
    embedder.embed.assert_called_once_with([c.text for c in chunks])


def test_ingest_calls_sparse_embed_with_chunk_texts():
    chunks = _make_chunks(3)
    service, _, _, sparse_embedder, _ = _make_service(chunks)
    service.ingest(Path("doc.pdf"), batch_size=10)
    sparse_embedder.embed_sparse.assert_called_once_with([c.text for c in chunks])


def test_ingest_upserts_all_points():
    service, *_, store = _make_service(_make_chunks(5))
    service.ingest(Path("doc.pdf"))
    total_upserted = sum(
        len(call_args.args[0]) for call_args in store.upsert.call_args_list
    )
    assert total_upserted == 5


def test_ingest_returns_zero_for_empty_pdf():
    service, parser, embedder, sparse_embedder, store = _make_service([])
    result = service.ingest(Path("empty.pdf"))
    assert result == 0
    embedder.embed.assert_not_called()
    sparse_embedder.embed_sparse.assert_not_called()
    store.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_ingest_splits_into_correct_batches():
    service, _, embedder, sparse_embedder, _ = _make_service(_make_chunks(10))
    service.ingest(Path("doc.pdf"), batch_size=3)
    # 10 chunks / batch_size 3 → 4 calls: [3, 3, 3, 1]
    assert embedder.embed.call_count == 4
    assert sparse_embedder.embed_sparse.call_count == 4
    sizes = [len(c.args[0]) for c in embedder.embed.call_args_list]
    assert sizes == [3, 3, 3, 1]


def test_ingest_single_batch_when_chunks_fit():
    service, _, embedder, *_ = _make_service(_make_chunks(5))
    service.ingest(Path("doc.pdf"), batch_size=100)
    assert embedder.embed.call_count == 1


# ---------------------------------------------------------------------------
# Vector structure
# ---------------------------------------------------------------------------


def test_ingest_point_has_dense_vector():
    chunks = _make_chunks(1)
    service, *_, store = _make_service(chunks)
    service.ingest(Path("doc.pdf"))
    points = store.upsert.call_args.args[0]
    assert "dense_vector" in points[0]


def test_ingest_point_has_sparse_vector():
    chunks = _make_chunks(1)
    service, *_, store = _make_service(chunks)
    service.ingest(Path("doc.pdf"))
    points = store.upsert.call_args.args[0]
    assert "sparse_vector" in points[0]
    assert isinstance(points[0]["sparse_vector"], SparseEmbedding)


def test_ingest_dense_vector_matches_embed_output():
    chunks = _make_chunks(2)
    service, *_, store = _make_service(chunks)
    service.ingest(Path("doc.pdf"))

    points = store.upsert.call_args.args[0]
    expected = _fake_embed([c.text for c in chunks])
    for point, vec in zip(points, expected):
        assert point["dense_vector"] == vec


# ---------------------------------------------------------------------------
# Payload structure
# ---------------------------------------------------------------------------


def test_ingest_payload_contains_required_fields():
    chunks = _make_chunks(1)
    service, *_, store = _make_service(chunks)
    service.ingest(Path("doc.pdf"), category="intro", version="v2")

    points = store.upsert.call_args.args[0]
    payload = points[0]["payload"]

    assert payload["text"] == chunks[0].text
    assert payload["source_file"] == "doc.pdf"
    assert payload["page"] == 0
    assert payload["chunk_index"] == 0
    assert payload["category"] == "intro"
    assert payload["version"] == "v2"


def test_ingest_payload_none_category_and_version_by_default():
    service, *_, store = _make_service(_make_chunks(1))
    service.ingest(Path("doc.pdf"))

    payload = store.upsert.call_args.args[0][0]["payload"]
    assert payload["category"] is None
    assert payload["version"] is None


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


def test_ingest_point_ids_are_deterministic():
    """Re-ingesting the same PDF must produce the same point IDs."""
    chunks = _make_chunks(3)
    service, _, _, _, store = _make_service(chunks)
    service.ingest(Path("doc.pdf"))

    ids_first = [p["id"] for p in store.upsert.call_args.args[0]]

    store.reset_mock()
    service.parser.parse.return_value = chunks
    service.ingest(Path("doc.pdf"))
    ids_second = [p["id"] for p in store.upsert.call_args.args[0]]

    assert ids_first == ids_second


def test_deterministic_id_formula():
    """ID must be a valid UUID derived from first 16 bytes of SHA-1."""
    import uuid
    chunk = TextChunk(text="x", source_file="a.pdf", page=2, chunk_index=5)
    digest_bytes = hashlib.sha1(b"a.pdf:2:5").digest()[:16]
    expected = str(uuid.UUID(bytes=digest_bytes))
    result = IngestionService._deterministic_id(chunk)
    assert result == expected
    uuid.UUID(result)  # raises ValueError if invalid


def test_different_chunks_produce_different_ids():
    c1 = TextChunk(text="x", source_file="a.pdf", page=0, chunk_index=0)
    c2 = TextChunk(text="x", source_file="a.pdf", page=0, chunk_index=1)
    assert IngestionService._deterministic_id(c1) != IngestionService._deterministic_id(c2)


# ---------------------------------------------------------------------------
# CLI — typer runner (no subprocess, no real files)
# ---------------------------------------------------------------------------


runner = CliRunner()


def _mock_dependencies(chunks: list[TextChunk], total: int = 3):
    """Return a set of patches that wire the CLI without real I/O."""
    mock_service = MagicMock()
    mock_service.ingest.return_value = total

    mock_embedder = MagicMock()
    mock_embedder.dim = 384

    mock_sparse_embedder = MagicMock()
    mock_sparse_embedder.embed_sparse.return_value = [SparseEmbedding(indices=[0], values=[1.0])]

    mock_store = MagicMock()

    return (
        patch("pdf_semantic_search.ingestion.cli.SentenceTransformerEmbedder", return_value=mock_embedder),
        patch("pdf_semantic_search.ingestion.cli.FastEmbedSparseEmbedder", return_value=mock_sparse_embedder),
        patch("pdf_semantic_search.ingestion.cli.QdrantStore", return_value=mock_store),
        patch("pdf_semantic_search.ingestion.cli.PyMuPDFParser"),
        patch("pdf_semantic_search.ingestion.cli.IngestionService", return_value=mock_service),
    )


def test_cli_succeeds_with_valid_pdf(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    p = _mock_dependencies(_make_chunks(3))
    with p[0], p[1], p[2], p[3], p[4]:
        result = runner.invoke(app, [str(pdf)])

    assert result.exit_code == 0, result.output
    assert "Done" in result.output or "chunk" in result.output


def test_cli_exits_nonzero_for_missing_file():
    result = runner.invoke(app, ["/nonexistent/totally/fake.pdf"])
    assert result.exit_code != 0


def test_cli_passes_chunk_size_to_service(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    p = _mock_dependencies(_make_chunks(1))
    with p[0], p[1], p[2], p[3], p[4] as mock_svc_cls:
        mock_svc = mock_svc_cls.return_value
        mock_svc.ingest.return_value = 1
        runner.invoke(app, [str(pdf), "--chunk-size", "256", "--overlap", "16"])

    mock_svc.ingest.assert_called_once()
    kwargs = mock_svc.ingest.call_args.kwargs
    assert kwargs["chunk_size"] == 256
    assert kwargs["overlap"] == 16


def test_cli_passes_category(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    p = _mock_dependencies(_make_chunks(1))
    with p[0], p[1], p[2], p[3], p[4] as mock_svc_cls:
        mock_svc = mock_svc_cls.return_value
        mock_svc.ingest.return_value = 1
        runner.invoke(app, [str(pdf), "--category", "finance"])

    kwargs = mock_svc.ingest.call_args.kwargs
    assert kwargs["category"] == "finance"


def test_cli_passes_version(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    p = _mock_dependencies(_make_chunks(1))
    with p[0], p[1], p[2], p[3], p[4] as mock_svc_cls:
        mock_svc = mock_svc_cls.return_value
        mock_svc.ingest.return_value = 1
        runner.invoke(app, [str(pdf), "--version", "v2"])

    kwargs = mock_svc.ingest.call_args.kwargs
    assert kwargs["version"] == "v2"


def test_cli_version_defaults_to_none(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    p = _mock_dependencies(_make_chunks(1))
    with p[0], p[1], p[2], p[3], p[4] as mock_svc_cls:
        mock_svc = mock_svc_cls.return_value
        mock_svc.ingest.return_value = 1
        runner.invoke(app, [str(pdf)])

    kwargs = mock_svc.ingest.call_args.kwargs
    assert kwargs["version"] is None
