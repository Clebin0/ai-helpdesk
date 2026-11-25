from __future__ import annotations

import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

EMBEDDING_DIMENSIONS = 384
MIGRATION_VERSION = "001_pgvector"
_migration_lock = threading.Lock()
_migrations_checked = False


def storage_backend() -> str:
    """Choose PostgreSQL when DATABASE_URL is configured; retain SQLite for legacy tests."""
    url = os.getenv("DATABASE_URL", "").strip().lower()
    if not url:
        return "sqlite"
    if url.startswith(("postgresql://", "postgres://")):
        return "postgres"
    raise RuntimeError("DATABASE_URL must use postgresql:// or postgres://.")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is required for the PostgreSQL storage backend.")
    if not url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("DATABASE_URL must use postgresql:// or postgres://.")
    return url


def _raw_connection():
    try:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL dependencies are missing. Install requirements.txt."
        ) from exc

    connection = psycopg.connect(
        _database_url(),
        connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
        row_factory=dict_row,
    )
    register_vector(connection)
    return connection


def _ensure_schema() -> None:
    global _migrations_checked
    if _migrations_checked:
        return

    with _migration_lock:
        if _migrations_checked:
            return
        migration_file = (
            Path(__file__).resolve().parent
            / "migrations"
            / f"{MIGRATION_VERSION}.sql"
        )
        if not migration_file.is_file():
            raise RuntimeError(f"Database migration not found: {migration_file}")

        with _raw_connection() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            applied = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (MIGRATION_VERSION,),
            ).fetchone()
            if not applied:
                statements = [
                    statement.strip()
                    for statement in migration_file.read_text(encoding="utf-8").split(";")
                    if statement.strip()
                ]
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (MIGRATION_VERSION,),
                )
        _migrations_checked = True


def _connection():
    _ensure_schema()
    return _raw_connection()


def check_database() -> bool:
    with _connection() as connection:
        row = connection.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)


def _validate_vectors(vectors: list[list[float]]) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding {index} has {len(vector)} dimensions; "
                f"the PostgreSQL schema requires {EMBEDDING_DIMENSIONS}."
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(f"Embedding {index} contains a non-finite value.")


def store_document(
    source: str,
    chunks: list[str],
    vectors: list[list[float]],
    model: str,
    digest: str,
    created_at: str,
) -> dict[str, Any]:
    from pgvector import Vector

    if len(chunks) != len(vectors):
        raise ValueError("The number of text chunks and embeddings must match.")
    _validate_vectors(vectors)
    created_at_value = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    with _connection() as connection:
        connection.execute(
            "DELETE FROM rag_chunks WHERE source = %s AND embedding_model = %s",
            (source, model),
        )
        connection.executemany(
            """
            INSERT INTO rag_chunks (
                source, chunk_index, content, embedding,
                embedding_model, content_sha256, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    source,
                    index,
                    chunk,
                    Vector(vector),
                    model,
                    digest,
                    created_at_value,
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors))
            ],
        )

    return {
        "source": source,
        "chunks": len(chunks),
        "embedding_model": model,
        "content_sha256": digest,
    }


def retrieve_chunks(
    query_vector: list[float], model: str, top_k: int
) -> list[dict[str, Any]]:
    from pgvector import Vector

    _validate_vectors([query_vector])
    limit = max(1, min(int(top_k), 8))
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT source, chunk_index, content,
                   1 - (embedding <=> %s) AS score
            FROM rag_chunks
            WHERE embedding_model = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (Vector(query_vector), model, Vector(query_vector), limit),
        ).fetchall()

    return [
        {
            "source": row["source"],
            "chunk_index": int(row["chunk_index"]),
            "content": row["content"],
            "score": round(float(row["score"]), 4),
        }
        for row in rows
    ]


def list_documents(model: str) -> list[dict[str, Any]]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT source, COUNT(*) AS chunks, MAX(created_at) AS updated_at
            FROM rag_chunks
            WHERE embedding_model = %s
            GROUP BY source
            ORDER BY source
            """,
            (model,),
        ).fetchall()

    return [
        {
            "source": row["source"],
            "chunks": int(row["chunks"]),
            "updated_at": (
                row["updated_at"].isoformat()
                if isinstance(row["updated_at"], datetime)
                else str(row["updated_at"])
            ),
        }
        for row in rows
    ]


def delete_document(source: str, model: str) -> None:
    with _connection() as connection:
        connection.execute(
            "DELETE FROM rag_chunks WHERE source = %s AND embedding_model = %s",
            (source, model),
        )
