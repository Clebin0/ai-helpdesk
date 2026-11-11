from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from typing import Any

TriageAgent = Callable[[dict[str, Any]], dict[str, Any]]
KnowledgeAgent = Callable[..., dict[str, Any]]

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def _review_citations(knowledge: dict[str, Any] | None) -> dict[str, Any]:
    """Validate citation references structurally; this is not semantic grounding."""
    if not knowledge:
        return {
            "status": "skipped",
            "method": "structural_only",
            "issues": [],
            "valid_citations": [],
        }

    sources = knowledge.get("sources") or []
    answer = str(knowledge.get("answer") or "")
    if not knowledge.get("used_llm") or not sources:
        return {
            "status": "not_applicable",
            "method": "structural_only",
            "issues": [],
            "valid_citations": [],
        }

    expected = {
        str(item.get("citation"))
        for item in sources
        if isinstance(item, dict) and item.get("citation")
    }
    found = set(_CITATION_PATTERN.findall(answer))
    unknown = sorted(found - expected)
    issues: list[str] = []
    if not found:
        issues.append("missing_citation")
    if unknown:
        issues.append("unknown_citation_reference")

    valid = sorted(found & expected)
    return {
        "status": "needs_human_review" if issues else "passed",
        "method": "structural_only",
        "issues": issues,
        "valid_citations": valid,
        "unknown_citations": unknown,
        "note": (
            "This check validates citation IDs only; it does not prove that each "
            "claim is semantically supported by the cited text."
        ),
    }


def run_agent_workflow(
    *,
    triage_agent: TriageAgent,
    knowledge_agent: KnowledgeAgent,
    question: str | None = None,
    ticket: dict[str, Any] | None = None,
    top_k: int = 4,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run bounded, in-process specialists. No shell, network, or write tools exist here."""
    normalized_question = question.strip() if isinstance(question, str) else ""
    if not normalized_question and ticket is None:
        raise ValueError("Provide a question, a ticket, or both.")
    if normalized_question and len(normalized_question) > 3000:
        raise ValueError("Question cannot exceed 3,000 characters.")
    if not 1 <= top_k <= 8:
        raise ValueError("top_k must be between 1 and 8.")
    if history is not None and len(history) > 8:
        raise ValueError("History cannot contain more than 8 turns.")

    selected_agents = ["coordinator"]
    triage_result: dict[str, Any] | None = None
    knowledge_result: dict[str, Any] | None = None

    if ticket is not None:
        selected_agents.append("triage")
        triage_result = triage_agent(ticket)

    if normalized_question:
        selected_agents.extend(["knowledge", "citation_integrity"])
        knowledge_result = knowledge_agent(
            normalized_question,
            top_k=top_k,
            history=history or [],
        )

    citation_review = _review_citations(knowledge_result)
    escalation_reasons: list[str] = []
    if triage_result and triage_result.get("human_review_required"):
        escalation_reasons.extend(
            str(reason)
            for reason in triage_result.get("human_review_reasons", [])
        )
        if not escalation_reasons:
            escalation_reasons.append("triage_requires_human_review")
    if citation_review["status"] == "needs_human_review":
        escalation_reasons.extend(citation_review["issues"])

    if escalation_reasons:
        status = "needs_human_review"
    elif knowledge_result and not knowledge_result.get("used_llm"):
        status = "abstained"
    else:
        status = "completed"

    return {
        "trace_id": uuid.uuid4().hex,
        "status": status,
        "selected_agents": selected_agents,
        "results": {
            "triage": triage_result,
            "knowledge": knowledge_result,
            "citation_integrity": citation_review,
        },
        "escalation": {
            "required": bool(escalation_reasons),
            "reasons": sorted(set(escalation_reasons)),
        },
        "capabilities": {
            "external_tools": False,
            "write_actions": False,
            "shell_access": False,
            "citation_review_is_semantic": False,
        },
    }
