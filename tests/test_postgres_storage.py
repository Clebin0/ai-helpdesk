from __future__ import annotations

import os
import uuid

import pytest

import assistant_storage


def test_storage_backend_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert assistant_storage.storage_backend() == "sqlite"


def test_storage_backend_detects_postgres(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://helpdesk:secret@localhost:5432/ai_helpdesk",
    )
    assert assistant_storage.storage_backend() == "postgres"


def test_storage_backend_rejects_unknown_scheme(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql://localhost/ai_helpdesk")
    with pytest.raises(RuntimeError, match="must use postgresql"):
        assistant_storage.storage_backend()


def test_postgres_vector_dimension_is_validated():
    with pytest.raises(ValueError, match="requires 384"):
        assistant_storage._validate_vectors([[0.0, 1.0]])


def test_postgres_vector_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        assistant_storage._validate_vectors([[float("nan")] * 384])


@pytest.mark.skipif(
    not os.getenv("RUN_POSTGRES_TESTS"),
    reason="PostgreSQL integration test runs in the GitHub Actions service container.",
)
def test_postgres_document_round_trip(monkeypatch):
    source = f"ci-integration-{uuid.uuid4().hex}.md"
    model = "test:deterministic-384"
    vector = [0.0] * 384
    vector[0] = 1.0
    monkeypatch.setenv("DATABASE_URL", os.environ["DATABASE_URL"])

    try:
        result = assistant_storage.store_document(
            source=source,
            chunks=["Documento de teste sobre diagnóstico de DNS."],
            vectors=[vector],
            model=model,
            digest="a" * 64,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert result["chunks"] == 1

        found = assistant_storage.retrieve_chunks(vector, model, 2)
        assert found
        assert found[0]["source"] == source
        assert found[0]["score"] == pytest.approx(1.0, abs=1e-5)

        documents = assistant_storage.list_documents(model)
        assert any(item["source"] == source and item["chunks"] == 1 for item in documents)
        assert assistant_storage.check_database() is True
    finally:
        assistant_storage.delete_document(source, model)
