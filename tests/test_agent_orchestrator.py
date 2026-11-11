from __future__ import annotations

import pytest

from agent_orchestrator import run_agent_workflow


def triage_clear(ticket):
    return {
        "category": "software",
        "priority": "low",
        "human_review_required": False,
        "human_review_reasons": [],
    }


def knowledge_with_citation(question, *, top_k, history):
    return {
        "answer": "Consulte o procedimento de DNS [S1].",
        "sources": [{"citation": "S1", "source": "network.md"}],
        "used_llm": True,
    }


def test_combined_workflow_routes_to_triage_and_knowledge():
    called = {"triage": 0, "knowledge": 0}

    def triage(ticket):
        called["triage"] += 1
        return triage_clear(ticket)

    def knowledge(question, *, top_k, history):
        called["knowledge"] += 1
        return knowledge_with_citation(question, top_k=top_k, history=history)

    result = run_agent_workflow(
        ticket={"title": "DNS falha", "description": "Falha para resolver nomes"},
        question="Como diagnosticar DNS?",
        triage_agent=triage,
        knowledge_agent=knowledge,
    )

    assert called == {"triage": 1, "knowledge": 1}
    assert result["status"] == "completed"
    assert result["selected_agents"] == [
        "coordinator",
        "triage",
        "knowledge",
        "citation_integrity",
    ]
    assert result["results"]["citation_integrity"]["valid_citations"] == ["S1"]
    assert result["capabilities"]["write_actions"] is False


def test_triage_escalation_requires_human_review():
    def triage(ticket):
        return {
            "category": "security",
            "priority": "critical",
            "human_review_required": True,
            "human_review_reasons": ["security_category", "critical_impact"],
        }

    result = run_agent_workflow(
        ticket={"title": "Suspeita de phishing"},
        triage_agent=triage,
        knowledge_agent=knowledge_with_citation,
    )

    assert result["status"] == "needs_human_review"
    assert result["escalation"]["required"] is True
    assert "security_category" in result["escalation"]["reasons"]


def test_missing_citation_escalates_instead_of_claiming_grounding():
    def knowledge(question, *, top_k, history):
        return {
            "answer": "Faça a alteração no servidor.",
            "sources": [{"citation": "S1", "source": "runbook.md"}],
            "used_llm": True,
        }

    result = run_agent_workflow(
        question="Como corrigir?",
        triage_agent=triage_clear,
        knowledge_agent=knowledge,
    )

    assert result["status"] == "needs_human_review"
    assert "missing_citation" in result["escalation"]["reasons"]
    assert result["results"]["citation_integrity"]["method"] == "structural_only"
    assert result["capabilities"]["citation_review_is_semantic"] is False


def test_unknown_citation_escalates():
    def knowledge(question, *, top_k, history):
        return {
            "answer": "Consulte a política [S9].",
            "sources": [{"citation": "S1", "source": "policy.md"}],
            "used_llm": True,
        }

    result = run_agent_workflow(
        question="Qual é a política?",
        triage_agent=triage_clear,
        knowledge_agent=knowledge,
    )

    assert result["status"] == "needs_human_review"
    assert "unknown_citation_reference" in result["escalation"]["reasons"]
    assert result["results"]["citation_integrity"]["unknown_citations"] == ["S9"]


def test_safe_rag_abstention_is_reported_as_abstained():
    def knowledge(question, *, top_k, history):
        return {
            "answer": "Não encontrei material suficiente.",
            "sources": [],
            "used_llm": False,
        }

    result = run_agent_workflow(
        question="Qual a política de VPN?",
        triage_agent=triage_clear,
        knowledge_agent=knowledge,
    )

    assert result["status"] == "abstained"
    assert result["escalation"]["required"] is False


def test_workflow_requires_question_or_ticket():
    with pytest.raises(ValueError, match="Provide a question"):
        run_agent_workflow(
            triage_agent=triage_clear,
            knowledge_agent=knowledge_with_citation,
        )


def test_workflow_rejects_invalid_top_k_and_oversized_history():
    with pytest.raises(ValueError, match="top_k"):
        run_agent_workflow(
            question="Como verificar DNS?",
            top_k=99,
            triage_agent=triage_clear,
            knowledge_agent=knowledge_with_citation,
        )

    with pytest.raises(ValueError, match="History"):
        run_agent_workflow(
            question="Como verificar DNS?",
            history=[{"role": "user", "content": "x"}] * 9,
            triage_agent=triage_clear,
            knowledge_agent=knowledge_with_citation,
        )
