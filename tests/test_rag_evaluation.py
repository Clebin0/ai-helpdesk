from __future__ import annotations

import pytest

from rag_evaluation import (
    audit_citations,
    evaluate_abstention,
    evaluate_citation_answers,
    evaluate_retrieval,
)


def test_retrieval_metrics_deduplicate_chunks_by_source():
    cases = [
        {
            "id": "dns",
            "question": "Como testar DNS?",
            "expected_sources": ["dns.md", "gateway.md"],
        }
    ]
    retrieved = {
        "dns": [
            {"source": "dns.md", "chunk_index": 0, "score": 0.99},
            {"source": "dns.md", "chunk_index": 1, "score": 0.95},
            {"source": "gateway.md", "chunk_index": 0, "score": 0.80},
        ]
    }

    result = evaluate_retrieval(cases, retrieved, top_k=2)

    assert result["recall_at_k"] == 1.0
    assert result["precision_at_k"] == 1.0
    assert result["precision_of_returned"] == 1.0
    assert result["average_sources_returned"] == 2.0
    assert result["mrr"] == 1.0
    assert result["hit_rate_at_k"] == 1.0
    assert result["per_query"][0]["retrieved_sources"] == ["dns.md", "gateway.md"]


def test_retrieval_metrics_penalize_relevant_source_ranked_second():
    cases = [{"id": "printer", "expected_sources": ["printer.md"]}]
    retrieved = {
        "printer": [
            {"source": "unrelated.md", "score": 0.9},
            {"source": "printer.md", "score": 0.8},
        ]
    }

    result = evaluate_retrieval(cases, retrieved, top_k=2)

    assert result["recall_at_k"] == 1.0
    assert result["precision_at_k"] == 0.5
    assert result["precision_of_returned"] == 0.5
    assert result["mrr"] == 0.5


def test_retrieval_metrics_return_zero_for_no_results():
    cases = [{"id": "dns", "expected_sources": ["dns.md"]}]
    result = evaluate_retrieval(cases, {"dns": []}, top_k=3)

    assert result["recall_at_k"] == 0.0
    assert result["precision_at_k"] == 0.0
    assert result["precision_of_returned"] == 0.0
    assert result["average_sources_returned"] == 0.0
    assert result["mrr"] == 0.0
    assert result["hit_rate_at_k"] == 0.0


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_retrieval_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        evaluate_retrieval(
            [{"id": "q1", "expected_sources": ["a.md"]}],
            {},
            top_k=top_k,
        )


def test_retrieval_rejects_duplicate_ids_and_empty_expected_sources():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_retrieval(
            [
                {"id": "q1", "expected_sources": ["a.md"]},
                {"id": "q1", "expected_sources": ["b.md"]},
            ],
            {},
        )

    with pytest.raises(ValueError, match="expected_sources"):
        evaluate_retrieval([{"id": "q2", "expected_sources": []}], {})


def test_citation_audit_checks_reference_integrity_and_source_alignment():
    result = audit_citations(
        "Confira o DNS [S1]. Veja também a fonte ausente [S9].",
        [
            {"citation": "S1", "source": "dns.md"},
            {"citation": "S2", "source": "gateway.md"},
        ],
        expected_sources=["dns.md"],
    )

    assert result["citation_count"] == 2
    assert result["valid_citation_count"] == 1
    assert result["citation_validity"] == 0.5
    assert result["relevant_citation_precision"] == 0.5
    assert result["relevant_source_recall"] == 1.0
    assert result["has_relevant_citation"] is True
    assert result["invalid_ids"] == ["S9"]


def test_citation_audit_handles_missing_citations():
    result = audit_citations(
        "Não há evidência suficiente para responder.",
        [{"citation": "S1", "source": "dns.md"}],
        expected_sources=["dns.md"],
    )

    assert result["citation_validity"] == 0.0
    assert result["relevant_citation_precision"] == 0.0
    assert result["relevant_source_recall"] == 0.0
    assert result["has_relevant_citation"] is False


def test_citation_answer_aggregation_requires_known_unique_queries():
    cases = [{"id": "dns", "expected_sources": ["dns.md"]}]
    answers = [
        {
            "query_id": "dns",
            "answer": "Use nslookup [S1].",
            "sources": [{"citation": "S1", "source": "dns.md"}],
        }
    ]

    result = evaluate_citation_answers(answers, cases)

    assert result["answer_count"] == 1
    assert result["citation_validity"] == 1.0
    assert result["relevant_citation_precision"] == 1.0
    assert result["relevant_source_recall"] == 1.0
    assert result["citation_coverage"] == 1.0
    assert result["semantic_entailment_evaluated"] is False

    with pytest.raises(ValueError, match="Unknown"):
        evaluate_citation_answers([{"query_id": "unknown"}], cases)

    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_citation_answers(answers * 2, cases)


def test_committed_benchmark_dataset_has_valid_expected_sources():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "evaluation" / "benchmark.json").read_text(encoding="utf-8"))
    sources = {document["source"] for document in dataset["documents"]}

    assert len(dataset["documents"]) == 8
    assert len(dataset["queries"]) == 21
    assert len({query["id"] for query in dataset["queries"]}) == len(dataset["queries"])
    assert all(query["answerable"] is True and query["expected_sources"] for query in dataset["queries"] if query["answerable"])
    assert all(query["answerable"] is False and not query["expected_sources"] for query in dataset["queries"] if not query["answerable"])
    assert all(set(query["expected_sources"]) <= sources for query in dataset["queries"])
    assert all(len(document["content"]) <= 500 for document in dataset["documents"])



def test_abstention_metrics_distinguish_correct_abstention_from_missed_answers():
    cases = [
        {"id":"dns","answerable":True,"expected_sources":["dns.md"]},
        {"id":"printer","answerable":True,"expected_sources":["printer.md"]},
        {"id":"vpn","answerable":False,"expected_sources":[]},
        {"id":"bios","answerable":False,"expected_sources":[]},
    ]
    retrieved = {
        "dns":[{"source":"dns.md","score":0.9}],
        "printer":[{"source":"gateway.md","score":0.8}],
        "vpn":[{"source":"network.md","score":0.7}],
        "bios":[],
    }

    result = evaluate_abstention(cases,retrieved,top_k=4)

    assert result["correct_abstentions"] == 1
    assert result["missed_answerable_queries"] == 0
    assert result["false_answers_on_unsupported_queries"] == 1
    assert result["answered_without_expected_evidence"] == 1
    assert result["answerability_accuracy"] == 0.75
    assert result["abstention_precision"] == 1.0
    assert result["abstention_recall"] == 0.5
    assert result["abstention_f1"] == pytest.approx(0.6667)
    assert result["answerable_retrieval_coverage"] == 0.5
    assert result["evidence_precision_among_answered_queries"] == pytest.approx(1/3)


def test_abstention_rejects_invalid_labels_and_empty_gold_sets():
    with pytest.raises(ValueError,match="boolean answerable"):
        evaluate_abstention([{"id":"q","expected_sources":[]}],{})
    with pytest.raises(ValueError,match="needs expected_sources"):
        evaluate_abstention([{"id":"q","answerable":True,"expected_sources":[]}],{})
    with pytest.raises(ValueError,match="must have empty expected_sources"):
        evaluate_abstention([{"id":"q","answerable":False,"expected_sources":["dns.md"]}],{})


def test_committed_benchmark_v2_has_canonical_paraphrase_and_unanswerable_cases():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "evaluation" / "benchmark.json").read_text(encoding="utf-8"))
    sources = {document["source"] for document in dataset["documents"]}
    cases = dataset["queries"]
    ids = {case["id"] for case in cases}

    assert dataset["version"] == 2
    assert len(dataset["documents"]) == 8
    assert len(cases) == 21
    assert len(ids) == len(cases)
    assert sum(case["type"] == "canonical" for case in cases) == 8
    assert sum(case["type"] == "paraphrase" for case in cases) == 8
    assert sum(case["type"] == "unanswerable" for case in cases) == 5
    assert all(case["answerable"] is True and case["expected_sources"] for case in cases if case["type"] != "unanswerable")
    assert all(case["answerable"] is False and not case["expected_sources"] for case in cases if case["type"] == "unanswerable")
    assert all(set(case["expected_sources"]) <= sources for case in cases)
    assert all(case["variant_of"] in ids for case in cases if case["type"] == "paraphrase")
