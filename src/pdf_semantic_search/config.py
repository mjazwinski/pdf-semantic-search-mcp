"""Centralised configuration loaded from environment / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    debug: bool = False
    log_path: str | None = None

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "pdf_chunks"

    # ── Dense embeddings ──────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Name of the dense vector field inside the Qdrant collection.
    dense_vector_name: str = "dense"

    # ── Sparse embeddings (SPLADE++ via fastembed, runs locally) ──────────────
    # Model is downloaded once on first use and cached locally.
    sparse_embedding_model: str = "prithivida/Splade_PP_en_v1"
    # Name of the sparse vector field inside the Qdrant collection.
    sparse_vector_name: str = "sparse"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64


# Singleton — import this everywhere
settings = Settings()
