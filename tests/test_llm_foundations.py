from __future__ import annotations

import json
from urllib.error import URLError

import pytest
from pydantic import ValidationError

from examples.llm_foundations.ollama_client import (
    MAX_RESPONSE_BYTES,
    OllamaClient,
    OllamaError,
)
from examples.llm_foundations.ticket_analysis import TicketAnalysis, analyze_ticket


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


def test_client_builds_non_streaming_chat_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = request.headers
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            json.dumps(
                {"message": {"role": "assistant", "content": "  Olá, mundo!  "}}
            ).encode("utf-8")
        )

    monkeypatch.setattr(
        "examples.llm_foundations.ollama_client.urlopen", fake_urlopen
    )
    client = OllamaClient(timeout=15)
    answer = client.chat(
        [
            {"role": "system", "content": "Responda em português."},
            {"role": "user", "content": "  Olá?  "},
        ]
    )

    assert answer == "Olá, mundo!"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 15
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["num_predict"] == 160
    assert captured["payload"]["messages"][1]["content"] == "Olá?"


def test_client_sends_json_schema_format(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            b'{"message":{"role":"assistant","content":"{\\"ok\\":true}"}}'
        )

    monkeypatch.setattr(
        "examples.llm_foundations.ollama_client.urlopen", fake_urlopen
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    client = OllamaClient()
    response = client.chat(
        [{"role": "user", "content": "Retorne JSON."}],
        format=schema,
        temperature=0,
    )

    assert json.loads(response) == {"ok": True}
    assert captured["payload"]["format"] == schema
    assert captured["payload"]["options"]["temperature"] == 0.0


@pytest.mark.parametrize(
    "host",
    [
        "ftp://127.0.0.1:11434",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434?token=secret",
        "http://127.0.0.1:99999",
        "",
    ],
)
def test_client_rejects_invalid_hosts(host):
    with pytest.raises(ValueError):
        OllamaClient(host=host)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_client_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError):
        OllamaClient(timeout=timeout)


def test_client_rejects_invalid_messages():
    invalid_messages = [
        [],
        "not a sequence of message objects",
        [{"role": ["user"], "content": "hello"}],
        [{"role": "tool", "content": "hello"}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": "x" * 4001}],
    ]

    for messages in invalid_messages:
        with pytest.raises(ValueError):
            OllamaClient._validate_messages(messages)


@pytest.mark.parametrize("temperature", [-0.1, 2.1, float("nan"), True, "0.2"])
def test_chat_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        OllamaClient().chat(
            [{"role": "user", "content": "Olá"}],
            temperature=temperature,
        )


@pytest.mark.parametrize("num_predict", [0, -1, 4097, True, 1.5])
def test_chat_rejects_invalid_output_limits(num_predict):
    with pytest.raises(ValueError):
        OllamaClient().chat(
            [{"role": "user", "content": "Olá"}],
            num_predict=num_predict,
        )


def test_chat_rejects_non_boolean_think_flag():
    with pytest.raises(ValueError):
        OllamaClient().chat(
            [{"role": "user", "content": "Olá"}],
            think="false",
        )


@pytest.mark.parametrize("format_value", [[], 12, "", {"properties": object()}])
def test_chat_rejects_invalid_format(monkeypatch, format_value):
    with pytest.raises(ValueError):
        OllamaClient().chat(
            [{"role": "user", "content": "Retorne JSON."}],
            format=format_value,
        )


def test_client_wraps_network_errors_without_leaking_transport_details(monkeypatch):
    def fail_urlopen(request, timeout):
        raise URLError("internal host and transport detail")

    monkeypatch.setattr(
        "examples.llm_foundations.ollama_client.urlopen", fail_urlopen
    )

    with pytest.raises(OllamaError, match="Não foi possível acessar o Ollama local"):
        OllamaClient().chat([{"role": "user", "content": "Olá"}])


def test_client_rejects_oversized_response(monkeypatch):
    monkeypatch.setattr(
        "examples.llm_foundations.ollama_client.urlopen",
        lambda request, timeout: FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1)),
    )

    with pytest.raises(OllamaError, match="limite de 1 MB"):
        OllamaClient().chat([{"role": "user", "content": "Olá"}])


def test_ticket_analysis_requests_schema_and_validates_output():
    captured = {}

    class FakeClient:
        def chat(self, messages, *, temperature, format):
            captured["messages"] = messages
            captured["temperature"] = temperature
            captured["format"] = format
            return json.dumps(
                {
                    "summary": "Falha de resolução DNS afeta o escritório.",
                    "category": "network",
                    "priority": "high",
                    "suggested_action": "Coletar IP, gateway e testar o resolvedor DNS.",
                    "requires_human_review": True,
                }
            )

    result = analyze_ticket(
        "Usuários do escritório não conseguem resolver nomes DNS.",
        client=FakeClient(),
    )

    assert isinstance(result, TicketAnalysis)
    assert result.category == "network"
    assert result.requires_human_review is True
    assert captured["temperature"] == 0.0
    assert captured["format"]["type"] == "object"
    assert captured["messages"][0]["role"] == "system"


def test_ticket_analysis_rejects_invalid_model_json():
    class FakeClient:
        def chat(self, messages, *, temperature, format):
            return '{"category": "made-up"}'

    with pytest.raises(ValidationError):
        analyze_ticket("Falha de rede no escritório.", client=FakeClient())


@pytest.mark.parametrize("ticket_text", ["", "   ", "x" * 4001, None])
def test_ticket_analysis_validates_input(ticket_text):
    with pytest.raises(ValueError):
        analyze_ticket(ticket_text)
