from __future__ import annotations

import pytest

from rag_e2e_evaluation import evaluate_end_to_end, select_balanced_cases


def _grounding(
    *,
    claims: int,
    supported: int = 0,
    unsupported: int = 0,
    insufficient: int = 0,
    cited: bool = False,
) -> dict:
    per_claim = []
    for index in range(claims):
        if index < supported:
            per_claim.append(
                {
                    "verdict": "supported",
                    "cited_supporting_ids": ["S1"] if cited else [],
                }
            )
        elif index < supported + unsupported:
            per_claim.append(
                {"verdict": "unsupported", "cited_supporting_ids": []}
            )
        else:
            per_claim.append(
                {"verdict": "insufficient_evidence", "cited_supporting_ids": []}
            )
    return {
        "claim_count": claims,
        "supported_claim_count": supported,
        "unsupported_claim_count": unsupported,
        "insufficient_evidence_claim_count": insufficient,
        "inconsistent_supported_claim_count": 0,
        "per_claim": per_claim,
    }


def test_end_to_end_metrics_cover_retrieval_generation_grounding_and_citations():
    cases = [
        {"id": "answerable-hit", "answerable": True, "expected_sources": ["dns.md"]},
        {"id": "answerable-miss", "answerable": True, "expected_sources": ["printer.md"]},
        {"id": "unsupported-hit", "answerable": False, "expected_sources": []},
        {"id": "unsupported-abstain", "answerable": False, "expected_sources": []},
    ]
    executions = [
        {
            "query_id": "answerable-hit",
            "answer": "Use nslookup para testar DNS [S1].",
            "retrieved_sources": [{"citation": "S1", "source": "dns.md"}],
            "used_llm": True,
            "grounding": _grounding(claims=1, supported=1, cited=True),
        },
        {
            "query_id": "answerable-miss",
            "answer": "Não encontrei conteúdo relevante.",
            "retrieved_sources": [],
            "used_llm": False,
            "grounding": None,
        },
        {
            "query_id": "unsupported-hit",
            "answer": "Reinicie o serviço de VPN [S1].",
            "retrieved_sources": [{"citation": "S1", "source": "dns.md"}],
            "used_llm": True,
            "grounding": _grounding(claims=1, unsupported=1),
        },
        {
            "query_id": "unsupported-abstain",
            "answer": "Não há informação suficiente na base.",
            "retrieved_sources": [],
            "used_llm": False,
            "grounding": None,
        },
    ]

    report = evaluate_end_to_end(cases, executions)

    assert report["query_count"] == 4
    assert report["retrieval"]["answerable_retrieval_coverage"] == 0.5
    assert report["retrieval"]["missed_answerable_queries"] == 1
    assert report["retrieval"]["correct_retriever_abstentions"] == 1
    assert report["retrieval"]["unsupported_query_abstention_rate"] == 0.5
    assert report["generation"]["llm_answer_count"] == 2
    assert report["generation"]["answerable_supported_claim_coverage"] == 0.5
    assert report["generation"]["unanswerable_safe_abstention_proxy_rate"] == 0.5
    assert report["generation"]["unanswerable_queries_with_factual_claims"] == 1
    assert report["grounding"]["claim_count"] == 2
    assert report["grounding"]["claim_support_rate"] == 0.5
    assert report["grounding"]["unsupported_claim_rate"] == 0.5
    assert report["grounding"]["citation_coverage_on_supported_claims"] == 1.0
    assert report["citations"]["citation_validity"] == 1.0
    assert report["citations"]["relevant_citation_precision"] == 0.5
    assert report["citations"]["answerable_query_citation_coverage"] == 0.5


def test_end_to_end_rejects_missing_duplicate_and_unknown_executions():
    cases = [
        {"id": "a", "answerable": True, "expected_sources": ["a.md"]},
        {"id": "u", "answerable": False, "expected_sources": []},
    ]
    empty_exec = {
        "query_id": "a",
        "answer": "Não encontrei conteúdo.",
        "retrieved_sources": [],
        "used_llm": False,
        "grounding": None,
    }
    with pytest.raises(ValueError, match="match benchmark cases"):
        evaluate_end_to_end(cases, [empty_exec])

    unsupported = {
        "query_id": "u",
        "answer": "Não há evidência.",
        "retrieved_sources": [],
        "used_llm": False,
        "grounding": None,
    }
    with pytest.raises(ValueError, match="Duplicate execution"):
        evaluate_end_to_end(cases, [empty_exec, unsupported, unsupported])

    unknown = {**unsupported, "query_id": "unknown"}
    with pytest.raises(ValueError, match="Unknown query_id"):
        evaluate_end_to_end(cases, [empty_exec, unknown])


def test_end_to_end_rejects_inconsistent_grounding_counts():
    cases = [
        {"id": "a", "answerable": True, "expected_sources": ["a.md"]},
        {"id": "u", "answerable": False, "expected_sources": []},
    ]
    base = {
        "answer": "Resposta.",
        "retrieved_sources": [],
        "used_llm": False,
        "grounding": None,
    }
    executions = [
        {
            **base,
            "query_id": "a",
            "grounding": _grounding(claims=2, supported=1),
        },
        {**base, "query_id": "u"},
    ]
    with pytest.raises(ValueError, match="grounding verdict totals do not match per_claim"):
        evaluate_end_to_end(cases, executions)


def test_end_to_end_requires_both_answerable_and_unanswerable_cases():
    cases = [{"id": "a", "answerable": True, "expected_sources": ["a.md"]}]
    execution = {
        "query_id": "a",
        "answer": "Não encontrei conteúdo.",
        "retrieved_sources": [],
        "used_llm": False,
        "grounding": None,
    }
    with pytest.raises(ValueError, match="need answerable and unanswerable"):
        evaluate_end_to_end(cases, [execution])


def test_balanced_selection_includes_canonical_paraphrase_and_unanswerable():
    cases = [
        {"id": "c1", "type": "canonical"},
        {"id": "c2", "type": "canonical"},
        {"id": "p1", "type": "paraphrase"},
        {"id": "p2", "type": "paraphrase"},
        {"id": "u1", "type": "unanswerable"},
        {"id": "u2", "type": "unanswerable"},
    ]

    selected = select_balanced_cases(cases, 6)

    assert [case["id"] for case in selected] == ["c1", "p1", "u1", "c2", "p2", "u2"]
    assert select_balanced_cases(cases, None) == cases
    with pytest.raises(ValueError):
        select_balanced_cases(cases, 0)


def test_judge_failure_is_recorded_and_not_counted_as_safe_abstention():
    cases = [
        {"id": "answerable", "answerable": True, "expected_sources": ["dns.md"]},
        {"id": "unsupported-with-context", "answerable": False, "expected_sources": []},
        {"id": "unsupported-no-context", "answerable": False, "expected_sources": []},
    ]
    executions = [
        {
            "query_id": "answerable",
            "answer": "Use nslookup [S1].",
            "retrieved_sources": [{"citation": "S1", "source": "dns.md"}],
            "used_llm": True,
            "grounding": None,
            "grounding_error": "ValueError: JSON incompleto",
        },
        {
            "query_id": "unsupported-with-context",
            "answer": "Use o procedimento de VPN [S1].",
            "retrieved_sources": [{"citation": "S1", "source": "dns.md"}],
            "used_llm": True,
            "grounding": None,
            "grounding_error": "ValueError: JSON incompleto",
        },
        {
            "query_id": "unsupported-no-context",
            "answer": "Não há informação suficiente na base.",
            "retrieved_sources": [],
            "used_llm": False,
            "grounding": None,
            "grounding_error": None,
        },
    ]

    report = evaluate_end_to_end(cases, executions)

    assert report["generation"]["grounding_evaluation_failure_count"] == 2
    assert report["generation"]["unanswerable_grounding_evaluation_failures"] == 1
    assert report["generation"]["grounding_evaluated_answer_count"] == 0
    assert report["generation"]["grounding_evaluation_coverage"] == 0.0
    assert report["generation"]["unanswerable_queries_without_factual_claims"] == 1
    assert report["generation"]["unanswerable_safe_abstention_proxy_rate"] == 0.5
    assert report["per_query"][1]["grounding_error"] == "ValueError: JSON incompleto"

