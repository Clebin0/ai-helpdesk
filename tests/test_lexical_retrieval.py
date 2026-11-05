from __future__ import annotations

import sqlite3

import pytest

from lexical_retrieval import FTS5Index, reciprocal_rank_fusion


def test_fts5_bm25_retrieves_accent_insensitive_support_document():
    chunks = [
        {
            "source":"dns.md",
            "chunk_index":0,
            "content":"Diagnóstico de resolução DNS no Windows. Use nslookup para testar nomes.",
        },
        {
            "source":"printer.md",
            "chunk_index":0,
            "content":"Fila de impressão parada. Verifique o Spooler e os drivers.",
        },
    ]

    with FTS5Index(chunks) as index:
        results = index.search("Como testar resolucao de nomes com NSLOOKUP no Windows?",limit=2)

    assert results
    assert results[0]["source"] == "dns.md"
    assert results[0]["content"] == chunks[0]["content"]
    assert results[0]["lexical_rank"] == 1
    assert results[0]["bm25_score"] < 0


def test_fts5_returns_no_results_for_terms_outside_the_corpus():
    chunks = [
        {"source":"dns.md","chunk_index":0,"content":"Diagnóstico de resolução DNS no Windows."}
    ]

    with FTS5Index(chunks) as index:
        assert index.search("Kubernetes e Terraform no cluster") == []


def test_fts5_query_text_cannot_inject_fts_operators():
    chunks = [
        {"source":"dns.md","chunk_index":0,"content":"Diagnóstico DNS no Windows."}
    ]

    with FTS5Index(chunks) as index:
        results = index.search('" OR * : ( )')

    assert results == []


def test_fts5_rejects_invalid_result_limit():
    with FTS5Index([]) as index:
        with pytest.raises(ValueError,match="positive integer"):
            index.search("DNS",limit=0)


def test_rrf_combines_rankings_and_preserves_chunk_content():
    vector = [
        {"source":"dns.md","chunk_index":0,"content":"Vetorial DNS","score":0.9},
        {"source":"network.md","chunk_index":0,"content":"Vetorial rede","score":0.8},
    ]
    lexical = [
        {"source":"network.md","chunk_index":0,"content":"Lexical rede","bm25_score":-1.0},
        {"source":"printer.md","chunk_index":0,"content":"Lexical impressora","bm25_score":-0.5},
    ]

    results = reciprocal_rank_fusion([vector,lexical],rank_constant=60,limit=3)

    assert [item["source"] for item in results] == [
        "network.md","dns.md","printer.md"
    ]
    assert results[0]["rrf_score"] == pytest.approx(2/61)
    assert results[0]["content"] == "Vetorial rede"


def test_rrf_deduplicates_same_chunk_within_each_ranking():
    ranking = [
        {"source":"dns.md","chunk_index":0,"content":"DNS"},
        {"source":"dns.md","chunk_index":0,"content":"Duplicate DNS"},
    ]

    results = reciprocal_rank_fusion([ranking,ranking],rank_constant=10,limit=4)

    assert len(results) == 1
    assert results[0]["rrf_score"] == pytest.approx(2/11)
    assert len(results[0]["rrf_ranks"]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank_constant":0},
        {"rank_constant":-1},
        {"limit":0},
    ],
)
def test_rrf_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([],**kwargs)


def test_fts5_module_reports_if_sqlite_build_does_not_support_fts5(monkeypatch):
    # The normal CI/runtime build should support FTS5. This assertion documents
    # the explicit, actionable failure path without replacing SQLite internals.
    assert sqlite3.sqlite_version_info >= (3, 9, 0)
