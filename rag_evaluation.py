from __future__ import annotations

import re
from statistics import mean
from typing import Any

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


def _unique_sources(chunks: list[dict[str, Any]]) -> list[str]:
    """Return source names in first-seen ranking order, without duplicate chunks."""
    ordered: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        source = str(chunk.get("source", "")).strip()
        if source and source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


def evaluate_retrieval(
    cases: list[dict[str, Any]],
    retrieved_by_query: dict[str, list[dict[str, Any]]],
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    """Calculate source-level Recall@K, Precision@K, MRR and HitRate@K.

    The production retriever returns chunks. This evaluator deduplicates chunks
    by source before scoring so a document split into multiple chunks cannot
    occupy several ranking positions.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    query_ids: set[str] = set()
    per_query: list[dict[str, Any]] = []

    for case in cases:
        query_id = str(case.get("id", "")).strip()
        if not query_id:
            raise ValueError("Every benchmark case must have a non-empty id.")
        if query_id in query_ids:
            raise ValueError(f"Duplicate benchmark query id: {query_id}")
        query_ids.add(query_id)

        expected = {
            str(source).strip()
            for source in case.get("expected_sources", [])
            if str(source).strip()
        }
        if not expected:
            raise ValueError(f"Benchmark case {query_id} has no expected_sources.")

        ranked = _unique_sources(retrieved_by_query.get(query_id, []))[:top_k]
        relevant_ranks = [
            rank for rank, source in enumerate(ranked, start=1)
            if source in expected
        ]
        hits = len(set(ranked) & expected)
        recall = hits / len(expected)
        precision = hits / top_k
        reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

        per_query.append(
            {
                "query_id": query_id,
                "expected_sources": sorted(expected),
                "retrieved_sources": ranked,
                "recall_at_k": round(recall, 4),
                "precision_at_k": round(precision, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
                "hit_at_k": bool(hits),
            }
        )

    if not per_query:
        raise ValueError("The benchmark must contain at least one query.")

    return {
        "top_k": top_k,
        "query_count": len(per_query),
        "recall_at_k": round(mean(item["recall_at_k"] for item in per_query), 4),
        "precision_at_k": round(mean(item["precision_at_k"] for item in per_query), 4),
        "mrr": round(mean(item["reciprocal_rank"] for item in per_query), 4),
        "hit_rate_at_k": round(mean(float(item["hit_at_k"]) for item in per_query), 4),
        "per_query": per_query,
    }


def audit_citations(
    answer: str,
    sources: list[dict[str, Any]],
    expected_sources: list[str] | set[str],
) -> dict[str, Any]:
    """Audit citation references against returned source metadata and gold sources.

    This checks reference integrity and source alignment, not whether the cited
    passage semantically entails every sentence in the answer.
    """
    cited_ids = list(dict.fromkeys(_CITATION_PATTERN.findall(answer or "")))
    source_by_citation = {
        str(item.get("citation", "")).upper(): str(item.get("source", "")).strip()
        for item in sources
        if item.get("citation") and item.get("source")
    }
    expected = {str(source).strip() for source in expected_sources if str(source).strip()}

    valid_ids = {citation for citation in cited_ids if citation in source_by_citation}
    relevant_ids = {
        citation
        for citation, source in source_by_citation.items()
        if source in expected
    }
    cited_relevant_ids = set(cited_ids) & relevant_ids

    citation_count = len(cited_ids)
    validity = len(valid_ids) / citation_count if citation_count else 0.0
    precision = len(cited_relevant_ids) / citation_count if citation_count else 0.0
    recall = (
        len({source_by_citation[citation] for citation in cited_relevant_ids}) / len(expected)
        if expected else 0.0
    )

    return {
        "cited_ids": cited_ids,
        "valid_citation_count": len(valid_ids),
        "citation_count": citation_count,
        "citation_validity": round(validity, 4),
        "relevant_citation_precision": round(precision, 4),
        "relevant_source_recall": round(recall, 4),
        "has_relevant_citation": bool(cited_relevant_ids),
        "invalid_ids": [citation for citation in cited_ids if citation not in source_by_citation],
    }


def evaluate_citation_answers(
    answers: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate offline citation audits for captured assistant answers."""
    case_by_id = {str(case["id"]): case for case in cases}
    seen: set[str] = set()
    audits: list[dict[str, Any]] = []

    for item in answers:
        query_id = str(item.get("query_id", "")).strip()
        if query_id not in case_by_id:
            raise ValueError(f"Unknown query_id in answers file: {query_id}")
        if query_id in seen:
            raise ValueError(f"Duplicate answer for query_id: {query_id}")
        seen.add(query_id)

        case = case_by_id[query_id]
        audit = audit_citations(
            str(item.get("answer", "")),
            item.get("sources", []),
            case.get("expected_sources", []),
        )
        audits.append({"query_id": query_id, **audit})

    if not audits:
        raise ValueError("The answers file must contain at least one answer.")

    return {
        "answer_count": len(audits),
        "citation_validity": round(mean(item["citation_validity"] for item in audits), 4),
        "relevant_citation_precision": round(
            mean(item["relevant_citation_precision"] for item in audits), 4
        ),
        "relevant_source_recall": round(mean(item["relevant_source_recall"] for item in audits), 4),
        "citation_coverage": round(
            mean(float(item["has_relevant_citation"]) for item in audits), 4
        ),
        "per_answer": audits,
        "semantic_entailment_evaluated": False,
    }
