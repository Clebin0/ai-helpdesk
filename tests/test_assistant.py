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
    monkeypatch.setenv("HF_TOKEN", "hf_test_not_real")
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local")
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


def test_embedding_model_namespaces_local_and_openai(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    assert assistant_core.embedding_model().startswith("fastembed:")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "openai")
    assert assistant_core.embedding_model() == "openai:text-embedding-3-small"


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
        lambda question, retrieved, history=None: "Verifique o resolvedor DNS e o gateway [S1].",
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
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf-super-secret-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-super-secret-key")
    response = client.get("/assistant/health")
    assert response.status_code == 200
    assert "hf-super-secret-key" not in response.text
    assert "openai-super-secret-key" not in response.text
    assert response.json()["llm_provider"] == "huggingface"
    assert response.json()["hf_token_configured"] is True
    assert response.json()["embedding_provider"] == "local"


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


def test_ingest_reports_exhausted_provider_credits(monkeypatch, tmp_path):
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
    assert "cota de inferência" in response.json()["detail"]


def test_chat_reports_exhausted_provider_credits(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)

    def fail_answer(question, top_k=4, history=None):
        raise FakeProviderQuotaError("provider quota exhausted")

    monkeypatch.setattr(assistant_core, "answer_question", fail_answer)
    response = client.post(
        "/assistant/chat",
        headers={"X-Assistant-Token": "test-chat-token"},
        json={"question": "Como diagnosticar DNS?"},
    )
    assert response.status_code == 402
    assert "cota de inferência" in response.json()["detail"]


def test_huggingface_chat_uses_openai_compatible_router(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_CHAT_MODEL", "test/model")

    class FakeMessage:
        content = "Confirme o resolvedor DNS [S1]."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def __init__(self):
            self.call = None

        def create(self, **kwargs):
            self.call = kwargs
            return FakeResponse()

    completions = FakeCompletions()

    class FakeClient:
        chat = type("Chat", (), {"completions": completions})()

    monkeypatch.setattr(assistant_core, "_huggingface_client", lambda: FakeClient())
    answer = assistant_core.generate_answer(
        "Como testar DNS?",
        [{"source": "dns.md", "chunk_index": 0, "content": "Teste o resolvedor DNS."}],
    )

    assert answer == "Confirme o resolvedor DNS [S1]."
    assert completions.call["model"] == "test/model"
    assert completions.call["max_tokens"] == 650
    assert completions.call["messages"][0]["role"] == "system"






def test_ollama_payload_disables_thinking_and_keeps_model_warm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b-q4_K_M")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "160")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")

    payload = assistant_core._ollama_payload(
        [{"role": "user", "content": "Responda objetivamente."}],
        stream=True,
    )

    assert payload["model"] == "qwen3.5:4b-q4_K_M"
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["options"]["num_ctx"] == 2048
    assert payload["options"]["num_predict"] == 160
    assert payload["options"]["num_thread"] == 6


def test_ollama_stream_parses_incremental_ndjson(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b-q4_K_M")
    captured = {}

    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            import json
            yield json.dumps({"message": {"content": "Primeiro "}, "done": False})
            yield json.dumps({"message": {"content": "segundo."}, "done": False})
            yield json.dumps({"message": {"content": ""}, "done": True})

    def fake_request(messages, stream):
        captured["messages"] = messages
        captured["stream"] = stream
        return FakeStreamResponse()

    monkeypatch.setattr(assistant_core, "_ollama_request", fake_request)
    history = [{"role": "user", "content": "Estou investigando DNS."}]
    chunks = list(
        assistant_core.generate_answer_stream(
            "E no Windows?",
            [{"source": "dns.md", "chunk_index": 0, "content": "Execute nslookup."}],
            history,
        )
    )

    assert chunks == ["Primeiro ", "segundo."]
    assert captured["stream"] is True
    assert "Estou investigando DNS." in captured["messages"][1]["content"]
    assert "E no Windows?" in captured["messages"][1]["content"]


def test_stream_endpoint_emits_metadata_tokens_and_done(monkeypatch, tmp_path):
    configure_test_environment(monkeypatch, tmp_path)
    relevant = [
        {
            "source": "network-runbook.md",
            "chunk_index": 0,
            "content": "Execute nslookup para validar a resolução DNS.",
            "score": 0.91,
        }
    ]
    received_history = []
    monkeypatch.setattr(assistant_core, "relevant_chunks", lambda question, top_k=4: relevant)
    monkeypatch.setattr(assistant_core, "source_metadata", lambda rows: [
        {"citation": "S1", "source": "network-runbook.md", "chunk_index": 0, "relevance": 0.91}
    ])

    def fake_stream(question, retrieved, history=None):
        received_history.extend(history or [])
        yield "Execute "
        yield "nslookup [S1]."

    monkeypatch.setattr(assistant_core, "generate_answer_stream", fake_stream)
    response = client.post(
        "/assistant/chat/stream",
        headers={"X-Assistant-Token": "test-chat-token"},
        json={
            "question": "Como testar DNS no Windows?",
            "history": [{"role": "user", "content": "O problema começou hoje."}],
        },
    )

    assert response.status_code == 200
    events = [__import__("json").loads(line) for line in response.text.splitlines() if line]
    assert [event["type"] for event in events] == ["meta", "token", "token", "done"]
    assert events[0]["sources"][0]["citation"] == "S1"
    assert events[1]["text"] + events[2]["text"] == "Execute nslookup [S1]."
    assert received_history[0]["content"] == "O problema começou hoje."


def test_stream_endpoint_requires_chat_token():
    response = client.post(
        "/assistant/chat/stream",
        json={"question": "Como diagnosticar DNS?"},
    )
    assert response.status_code == 401


def test_chat_ui_route_serves_interface():
    response = client.get("/assistant/ui")
    assert response.status_code == 200
    assert "Copiloto de suporte" in response.text
    assert "/assistant/chat/stream" in response.text
    assert "sessionStorage" in response.text
