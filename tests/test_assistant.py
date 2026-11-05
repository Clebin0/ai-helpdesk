from __future__ import annotations

import math

from fastapi.testclient import TestClient

import assistant_core
from app import app

client = TestClient(app)


def deterministic_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        normalized = text.casefold()
        vectors.append(
            [
                float("dns" in normalized or "network" in normalized),
                float("printer" in normalized or "printing" in normalized),
                float("password" in normalized or "access" in normalized),
            ]
        )
    return vectors


def configure_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "assistant.sqlite3"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ASSISTANT_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("ASSISTANT_ACCESS_TOKEN", "test-chat-token")
    monkeypatch.setattr(assistant_core, "embed_texts", deterministic_embeddings)


def test_chunk_text_handles_empty_and_long_documents():
    assert assistant_core.chunk_text("   ") == []
    chunks = assistant_core.chunk_text(("DNS troubleshooting.\n" * 500))
    assert len(chunks) > 1
    assert all(chunks)


def test_cosine_similarity_handles_zero_and_mismatched_vectors():
    assert assistant_core._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert assistant_core._cosine_similarity([1.0], [1.0, 0.0]) == 0.0
    assert math.isclose(
        assistant_core._cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0
    )


def test_ingestion_is_idempotent_for_same_source(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)
    first = assistant_core.ingest_document(
        "network-runbook.md",
        "DNS troubleshooting: verify resolver and gateway before changing settings.",
    )
    second = assistant_core.ingest_document(
        "network-runbook.md",
        "DNS troubleshooting: verify resolver and gateway before changing settings.",
    )
    assert first["chunks"] == 1
    assert second["content_sha256"] == first["content_sha256"]
    assert len(assistant_core.list_documents()) == 1
    assert assistant_core.list_documents()[0]["chunks"] == 1


def test_retrieval_ranks_relevant_document_first(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)
    assistant_core.ingest_document(
        "network.md", "DNS network resolver and gateway diagnostics."
    )
    assistant_core.ingest_document(
        "printing.md", "Printer queue and printing driver diagnostics."
    )
    results = assistant_core.retrieve_chunks("DNS network issue", top_k=2)
    assert results[0]["source"] == "network.md"
    assert results[0]["score"] > results[1]["score"]


def test_chat_returns_safe_fallback_when_retrieval_is_weak(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)
    assistant_core.ingest_document("printing.md", "Printer driver queue troubleshooting.")
    result = assistant_core.answer_question("What is the company VPN recovery policy?")
    assert result["used_llm"] is False
    assert result["sources"] == []
    assert "Não encontrei" in result["answer"]


def test_api_requires_admin_and_chat_tokens(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)
    no_admin = client.post(
        "/assistant/ingest",
        json={"source": "network.md", "content": "DNS and network diagnostic guide."},
    )
    no_chat = client.post("/assistant/chat", json={"question": "How do I verify DNS?"})
    assert no_admin.status_code == 401
    assert no_chat.status_code == 401


def test_api_ingest_and_chat_return_cited_sources(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        assistant_core,
        "generate_answer",
        lambda question, retrieved: "Verifique o resolvedor DNS e o gateway [S1].",
    )
    ingested = client.post(
        "/assistant/ingest",
        headers={"X-Admin-Token": "test-admin-token"},
        json={
            "source": "network-runbook.md",
            "content": "DNS diagnostics: verify the resolver, gateway and affected network.",
        },
    )
    assert ingested.status_code == 200
    assert ingested.json()["status"] == "indexed"

    response = client.post(
        "/assistant/chat",
        headers={"X-Assistant-Token": "test-chat-token"},
        json={"question": "How do I troubleshoot DNS on the network?"},
    )
    assert response.status_code == 200
    assert response.json()["used_llm"] is True
    assert response.json()["sources"][0]["citation"] == "S1"
    assert "[S1]" in response.json()["answer"]


def test_health_does_not_expose_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")
    response = client.get("/assistant/health")
    assert response.status_code == 200
    assert "super-secret-key" not in response.text
    assert response.json()["openai_configured"] is True


def test_api_rejects_extra_fields():
    response = client.post(
        "/assistant/chat",
        headers={"X-Assistant-Token": "test-chat-token"},
        json={"question": "How do I troubleshoot DNS?", "raw_api_key": "no"},
    )
    assert response.status_code == 422


class FakeProviderResponse:
    def json(self):
        return {
            "error": {
                "code": "credit_balance_exhausted",
                "type": "insufficient_quota",
            }
        }


class FakeProviderQuotaError(Exception):
    status_code = 429
    response = FakeProviderResponse()


def test_ingest_reports_exhausted_openai_credits(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)

    def fail_ingestion(source, content):
        raise FakeProviderQuotaError("provider quota exhausted")

    monkeypatch.setattr(assistant_core, "ingest_document", fail_ingestion)
    response = client.post(
        "/assistant/ingest",
        headers={"X-Admin-Token": "test-admin-token"},
        json={
            "source": "network-runbook.md",
            "content": "DNS diagnostics: verify the resolver and gateway.",
        },
    )
    assert response.status_code == 402
    assert "sem créditos disponíveis" in response.json()["detail"]


def test_chat_reports_exhausted_openai_credits(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)

    def fail_answer(question, top_k=4):
        raise FakeProviderQuotaError("provider quota exhausted")

    monkeypatch.setattr(assistant_core, "answer_question", fail_answer)
    response = client.post(
        "/assistant/chat",
        headers={"X-Assistant-Token": "test-chat-token"},
        json={"question": "Como diagnosticar DNS?"},
    )
    assert response.status_code == 402
    assert "sem créditos disponíveis" in response.json()["detail"]
