from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module
import assistant_core

client = TestClient(app_module.app)


def test_agents_endpoint_requires_existing_assistant_token(monkeypatch):
    monkeypatch.delenv("ASSISTANT_ACCESS_TOKEN", raising=False)
    response = client.post(
        "/agents/assist",
        json={"question": "Como diagnosticar DNS?"},
    )
    assert response.status_code == 503

    monkeypatch.setenv("ASSISTANT_ACCESS_TOKEN", "agent-test-token")
    denied = client.post(
        "/agents/assist",
        headers={"X-Assistant-Token": "wrong-token"},
        json={"question": "Como diagnosticar DNS?"},
    )
    assert denied.status_code == 401


def test_agents_endpoint_runs_knowledge_specialist_and_returns_citation_review(
    monkeypatch,
):
    monkeypatch.setenv("ASSISTANT_ACCESS_TOKEN", "agent-test-token")
    monkeypatch.setattr(
        assistant_core,
        "answer_question",
        lambda question, top_k=4, history=None: {
            "answer": "Verifique o resolvedor DNS [S1].",
            "sources": [{"citation": "S1", "source": "network.md"}],
            "used_llm": True,
        },
    )

    response = client.post(
        "/agents/assist",
        headers={"X-Assistant-Token": "agent-test-token"},
        json={"question": "Como diagnosticar DNS?", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert "knowledge" in payload["selected_agents"]
    assert payload["results"]["citation_integrity"]["status"] == "passed"
    assert payload["capabilities"]["external_tools"] is False


def test_agents_endpoint_escalates_ambiguous_triage(monkeypatch):
    monkeypatch.setenv("ASSISTANT_ACCESS_TOKEN", "agent-test-token")
    monkeypatch.setattr(
        app_module,
        "triage_one",
        lambda ticket: {
            "category": "security",
            "priority": "critical",
            "human_review_required": True,
            "human_review_reasons": ["security_category", "critical_impact"],
        },
    )

    response = client.post(
        "/agents/assist",
        headers={"X-Assistant-Token": "agent-test-token"},
        json={
            "ticket": {
                "title": "Suspeita de phishing",
                "description": "Usuário recebeu link suspeito.",
                "affected_users": 1,
                "service_stopped": False,
                "vip_or_critical_area": False,
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_human_review"
    assert payload["escalation"]["required"] is True
    assert "security_category" in payload["escalation"]["reasons"]


def test_agents_endpoint_requires_question_or_ticket(monkeypatch):
    monkeypatch.setenv("ASSISTANT_ACCESS_TOKEN", "agent-test-token")
    response = client.post(
        "/agents/assist",
        headers={"X-Assistant-Token": "agent-test-token"},
        json={},
    )
    assert response.status_code == 422
