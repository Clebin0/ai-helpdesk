"""End-to-end metrics for retrieval, generated answers, grounding and abstention."""

from __future__ import annotations

from statistics import mean
from typing import Any

from rag_evaluation import audit_citations


def _source_names(items: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each retrieved source must be an object.")
        source = str(item.get("source", "")).strip()
        if not source:
            raise ValueError("Each retrieved source must define source.")
        if source not in seen:
            seen.add(source)
            names.append(source)
    return names


def _non_negative_count(report: dict[str, Any] | None, key: str) -> int:
    if report is None:
        return 0
    value = report.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"grounding.{key} must be a non-negative integer.")
    return value


def evaluate_end_to_end(
    cases: list[dict[str, Any]],
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score real benchmark executions without treating the LLM judge as ground truth.

    Each execution contains query_id, answer, retrieved source metadata, used_llm,
    and an optional report created by summarize_grounding().
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list.")
    if not isinstance(executions, list) or not executions:
        raise ValueError("executions must be a non-empty list.")

    case_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each benchmark case must be an object.")
        query_id = str(case.get("id", "")).strip()
        if not query_id or query_id in case_by_id:
            raise ValueError(f"Missing or duplicate benchmark query id: {query_id!r}")
        answerable = case.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"Benchmark case {query_id} must define boolean answerable.")
        expected = case.get("expected_sources", [])
        if not isinstance(expected, list):
            raise ValueError(f"Benchmark case {query_id} expected_sources must be a list.")
        expected_names = {str(source).strip() for source in expected if str(source).strip()}
        if answerable and not expected_names:
            raise ValueError(f"Answerable case {query_id} needs expected_sources.")
        if not answerable and expected_names:
            raise ValueError(f"Unanswerable case {query_id} must have empty expected_sources.")
        case_by_id[query_id] = case

    execution_by_id: dict[str, dict[str, Any]] = {}
    for execution in executions:
        if not isinstance(execution, dict):
            raise ValueError("Each execution must be an object.")
        query_id = str(execution.get("query_id", "")).strip()
        if query_id not in case_by_id:
            raise ValueError(f"Unknown query_id in executions: {query_id}")
        if query_id in execution_by_id:
            raise ValueError(f"Duplicate execution for query_id: {query_id}")
        if not isinstance(execution.get("answer"), str):
            raise ValueError(f"Execution {query_id} must contain answer as string.")
        if not isinstance(execution.get("retrieved_sources"), list):
            raise ValueError(f"Execution {query_id} retrieved_sources must be a list.")
        if not isinstance(execution.get("used_llm"), bool):
            raise ValueError(f"Execution {query_id} must define boolean used_llm.")
        citations: set[str] = set()
        for source in execution["retrieved_sources"]:
            if not isinstance(source, dict):
                raise ValueError(f"Execution {query_id} contains an invalid source.")
            citation = str(source.get("citation", "")).strip().upper()
            if not citation or citation in citations:
                raise ValueError(f"Execution {query_id} has a missing or duplicate citation ID.")
            citations.add(citation)
        grounding = execution.get("grounding")
        if grounding is not None and not isinstance(grounding, dict):
            raise ValueError(f"Execution {query_id} grounding must be an object or null.")
        if execution["used_llm"] and grounding is None:
            raise ValueError(f"Execution {query_id} needs grounding when used_llm is true.")
        execution_by_id[query_id] = execution

    missing = set(case_by_id) - set(execution_by_id)
    extra = set(execution_by_id) - set(case_by_id)
    if missing or extra:
        raise ValueError(
            f"Executions must match benchmark cases; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )

    answerable_count = sum(bool(case["answerable"]) for case in cases)
    unanswerable_count = len(cases) - answerable_count
    if not answerable_count or not unanswerable_count:
        raise ValueError("The selected benchmark cases need answerable and unanswerable queries.")

    answerable_with_evidence = 0
    missed_answerable = 0
    unsupported_retrievals = 0
    correct_retriever_abstentions = 0
    answerable_with_supported_claims = 0
    unanswerable_without_claims = 0
    unanswerable_with_claims = 0
    total_claims = 0
    supported_claims = 0
    unsupported_claims = 0
    insufficient_claims = 0
    inconsistent_supported_claims = 0
    claims_with_verified_citations = 0
    total_citations = 0
    valid_citations = 0
    relevant_citations = 0
    queries_with_relevant_citation = 0
    relevant_source_recalls: list[float] = []
    llm_answer_count = 0
    per_query: list[dict[str, Any]] = []

    for case in cases:
        query_id = str(case["id"])
        execution = execution_by_id[query_id]
        answerable = case["answerable"]
        expected = {str(source).strip() for source in case.get("expected_sources", []) if str(source).strip()}
        sources = execution["retrieved_sources"]
        retrieved_names = _source_names(sources)
        retrieved_set = set(retrieved_names)
        expected_retrieved = sorted(retrieved_set & expected)
        has_expected_evidence = bool(expected_retrieved)
        has_context = bool(retrieved_names)

        if answerable:
            if has_expected_evidence:
                answerable_with_evidence += 1
            else:
                missed_answerable += 1
        elif has_context:
            unsupported_retrievals += 1
        else:
            correct_retriever_abstentions += 1

        answer = execution["answer"]
        citation_audit = audit_citations(answer, sources, expected)
        total_citations += citation_audit["citation_count"]
        valid_citations += citation_audit["valid_citation_count"]
        relevant_citations += round(
            citation_audit["relevant_citation_precision"] * citation_audit["citation_count"]
        )
        relevant_source_recalls.append(citation_audit["relevant_source_recall"])
        if answerable and citation_audit["has_relevant_citation"]:
            queries_with_relevant_citation += 1

        grounding = execution.get("grounding")
        claim_count = _non_negative_count(grounding, "claim_count")
        supported = _non_negative_count(grounding, "supported_claim_count")
        unsupported = _non_negative_count(grounding, "unsupported_claim_count")
        insufficient = _non_negative_count(grounding, "insufficient_evidence_claim_count")
        inconsistent = _non_negative_count(grounding, "inconsistent_supported_claim_count")
        verified_citation_claims = 0
        if grounding is not None:
            claim_details = grounding.get("per_claim", [])
            if not isinstance(claim_details, list) or len(claim_details) != claim_count:
                raise ValueError(f"Execution {query_id} has invalid per_claim details.")
            verified_citation_claims = sum(
                1
                for claim in claim_details
                if isinstance(claim, dict)
                and claim.get("verdict") == "supported"
                and bool(claim.get("cited_supporting_ids"))
            )
        if supported + unsupported + insufficient != claim_count:
            raise ValueError(f"Execution {query_id} has inconsistent grounding claim totals.")

        total_claims += claim_count
        supported_claims += supported
        unsupported_claims += unsupported
        insufficient_claims += insufficient
        inconsistent_supported_claims += inconsistent
        claims_with_verified_citations += verified_citation_claims
        if execution["used_llm"]:
            llm_answer_count += 1

        if answerable and supported > 0:
            answerable_with_supported_claims += 1
        if not answerable:
            if claim_count == 0:
                unanswerable_without_claims += 1
            else:
                unanswerable_with_claims += 1

        per_query.append(
            {
                "query_id": query_id,
                "answerable": answerable,
                "expected_sources": sorted(expected),
                "retrieved_sources": retrieved_names,
                "retrieved_expected_sources": expected_retrieved,
                "retrieved_expected_evidence": has_expected_evidence,
                "retriever_abstained": not has_context,
                "used_llm": execution["used_llm"],
                "answer": answer,
                "claim_count": claim_count,
                "supported_claim_count": supported,
                "unsupported_claim_count": unsupported,
                "insufficient_evidence_claim_count": insufficient,
                "citation_audit": citation_audit,
                "grounding": grounding,
            }
        )

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    citation_validity = ratio(valid_citations, total_citations)
    citation_precision = ratio(relevant_citations, total_citations)
    claim_support_rate = ratio(supported_claims, total_claims)
    unsupported_claim_rate = ratio(unsupported_claims, total_claims)
    insufficient_claim_rate = ratio(insufficient_claims, total_claims)

    return {
        "query_count": len(cases),
        "answerable_query_count": answerable_count,
        "unanswerable_query_count": unanswerable_count,
        "retrieval": {
            "answerable_retrieval_coverage": ratio(answerable_with_evidence, answerable_count),
            "missed_answerable_queries": missed_answerable,
            "correct_retriever_abstentions": correct_retriever_abstentions,
            "unsupported_queries_with_retrieved_context": unsupported_retrievals,
            "unsupported_query_abstention_rate": ratio(
                correct_retriever_abstentions, unanswerable_count
            ),
        },
        "generation": {
            "llm_answer_count": llm_answer_count,
            "answerable_queries_with_supported_claims": answerable_with_supported_claims,
            "answerable_supported_claim_coverage": ratio(
                answerable_with_supported_claims, answerable_count
            ),
            "unanswerable_queries_without_factual_claims": unanswerable_without_claims,
            "unanswerable_safe_abstention_proxy_rate": ratio(
                unanswerable_without_claims, unanswerable_count
            ),
            "unanswerable_queries_with_factual_claims": unanswerable_with_claims,
            "note": (
                "No factual claims is a conservative abstention proxy, not proof that "
                "the natural-language answer is a correct refusal."
            ),
        },
        "grounding": {
            "claim_count": total_claims,
            "supported_claim_count": supported_claims,
            "unsupported_claim_count": unsupported_claims,
            "insufficient_evidence_claim_count": insufficient_claims,
            "inconsistent_supported_claim_count": inconsistent_supported_claims,
            "claim_support_rate": claim_support_rate,
            "unsupported_claim_rate": unsupported_claim_rate,
            "insufficient_evidence_rate": insufficient_claim_rate,
            "citation_coverage_on_supported_claims": ratio(
                claims_with_verified_citations, supported_claims
            ),
        },
        "citations": {
            "citation_count": total_citations,
            "valid_citation_count": valid_citations,
            "invalid_citation_count": total_citations - valid_citations,
            "citation_validity": citation_validity,
            "relevant_citation_precision": citation_precision,
            "mean_relevant_source_recall": round(mean(relevant_source_recalls), 4),
            "answerable_query_citation_coverage": ratio(
                queries_with_relevant_citation, answerable_count
            ),
        },
        "per_query": per_query,
        "scope": (
            "end_to_end_local_evaluation; retrieval and citation integrity are deterministic "
            "against the synthetic benchmark; claim grounding is estimated by an LLM judge "
            "and requires human review"
        ),
    }


def select_balanced_cases(
    cases: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced smoke-test sample across case types."""
    if limit is None:
        return list(cases)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    preferred_types = ("canonical", "paraphrase", "unanswerable")
    buckets = {
        case_type: [case for case in cases if case.get("type") == case_type]
        for case_type in preferred_types
    }
    selected: list[dict[str, Any]] = []
    positions = {case_type: 0 for case_type in preferred_types}
    while len(selected) < min(limit, len(cases)):
        progressed = False
        for case_type in preferred_types:
            position = positions[case_type]
            if position < len(buckets[case_type]) and len(selected) < limit:
                selected.append(buckets[case_type][position])
                positions[case_type] += 1
                progressed = True
        if not progressed:
            remaining = [
                case for case in cases
                if case.get("type") not in preferred_types and case not in selected
            ]
            if not remaining:
                break
            selected.append(remaining[0])
    return selected
