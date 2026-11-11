from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.evaluate_rag_e2e as runner


class FakeAssistantCore:
    def __init__(self):
        self.semantic_results = [{"source": "semantic.md", "score": 0.2}]
        self.dense_results = [{"source": "dense.md", "score": 0.8}]
        self.relevant_call = None
        self.retrieve_call = None

    def relevant_chunks(self, question: str, *, top_k: int):
        self.relevant_call = (question, top_k)
        return self.semantic_results

    def retrieve_chunks(self, question: str, *, top_k: int):
        self.retrieve_call = (question, top_k)
        return self.dense_results


def _args(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval_mode=mode,
        top_k=4,
        rrf_candidate_depth=8,
        hybrid_semantic_weight=0.8,
        hybrid_min_lexical_coverage=0.1,
        hybrid_semantic_rescue=0.55,
        rrf_rank_constant=60,
        rrf_bm25_cutoff=-2.5,
    )


def test_semantic_mode_keeps_the_existing_production_retriever():
    assistant = FakeAssistantCore()

    result = runner._retrieve_chunks(
        "falha de DNS",
        assistant_core=assistant,
        args=_args("semantic"),
        minimum_relevance=0.18,
    )

    assert result == assistant.semantic_results
    assert assistant.relevant_call == ("falha de DNS", 4)
    assert assistant.retrieve_call is None


def test_hybrid_mode_filters_weak_reranked_candidates(monkeypatch):
    assistant = FakeAssistantCore()
    reranked = [
        {"source": "relevant.md", "score": 0.31},
        {"source": "weak.md", "score": 0.17},
    ]
    observed = {}

    def fake_rerank(question, candidates, **kwargs):
        observed["question"] = question
        observed["candidates"] = candidates
        observed["kwargs"] = kwargs
        return reranked

    monkeypatch.setattr(runner, "rerank_hybrid", fake_rerank)

    result = runner._retrieve_chunks(
        "configurar DNS",
        assistant_core=assistant,
        args=_args("hybrid"),
        minimum_relevance=0.18,
    )

    assert result == [reranked[0]]
    assert assistant.retrieve_call == ("configurar DNS", 8)
    assert observed["question"] == "configurar DNS"
    assert observed["candidates"] == assistant.dense_results
    assert observed["kwargs"] == {
        "semantic_weight": 0.8,
        "minimum_lexical_coverage": 0.1,
        "semantic_rescue_threshold": 0.55,
    }


def test_rrf_mode_filters_bm25_candidates_and_fuses_rankings(monkeypatch):
    assistant = FakeAssistantCore()
    vector_candidates = [{"source": "vector.md", "score": 0.31}]
    monkeypatch.setattr(
        runner,
        "rerank_hybrid",
        lambda question, candidates, **kwargs: vector_candidates,
    )

    lexical = SimpleNamespace(
        search=lambda question, *, limit: [
            {"source": "lexical.md", "bm25_score": -3.1},
            {"source": "weak-lexical.md", "bm25_score": -1.2},
        ]
    )
    observed = {}

    def fake_fusion(rankings, *, rank_constant, limit):
        observed["rankings"] = rankings
        observed["rank_constant"] = rank_constant
        observed["limit"] = limit
        return [{"source": "fused.md", "score": 0.01}]

    monkeypatch.setattr(runner, "reciprocal_rank_fusion", fake_fusion)

    result = runner._retrieve_chunks(
        "configurar DNS",
        assistant_core=assistant,
        args=_args("rrf"),
        minimum_relevance=0.18,
        lexical_index=lexical,
    )

    assert result == [{"source": "fused.md", "score": 0.01}]
    assert observed["rankings"] == [
        vector_candidates,
        [{"source": "lexical.md", "bm25_score": -3.1}],
    ]
    assert observed["rank_constant"] == 60
    assert observed["limit"] == 4


def test_rrf_mode_requires_its_lexical_index():
    assistant = FakeAssistantCore()

    with pytest.raises(RuntimeError, match="FTS5 index was not initialized"):
        runner._retrieve_chunks(
            "configurar DNS",
            assistant_core=assistant,
            args=_args("rrf"),
            minimum_relevance=0.18,
        )
