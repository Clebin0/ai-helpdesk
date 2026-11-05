from __future__ import annotations

import pytest

from hybrid_retrieval import rerank_hybrid, tokenize


def test_tokenize_normalizes_accents_and_hyphenated_technical_terms():
    assert tokenize("Falha de e-mail e Wi-Fi na conexão") == [
        "falha", "email", "wifi", "conexao"
    ]
    assert tokenize("Recebi link e anexos suspeitos em vários sites") == [
        "recebi", "link", "anexo", "suspeita", "varios", "site"
    ]
    assert "restaurar" in tokenize(
        "Preciso trazer um arquivo de volta usando um backup"
    )
    assert "restaurar" in tokenize("Restauração de backup")


def test_hybrid_reranker_prefers_lexically_supported_candidate():
    chunks = [
        {"source":"generic.md","content":"Suporte técnico e atendimento ao usuário.","score":0.68},
        {"source":"dns.md","content":"Diagnóstico DNS no Windows. Use nslookup e ipconfig para testar resolução DNS.","score":0.61},
    ]

    ranked = rerank_hybrid("Como testar DNS no Windows com nslookup?",chunks)

    assert ranked[0]["source"] == "dns.md"
    assert ranked[0]["lexical_coverage"] > ranked[1]["lexical_coverage"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_hybrid_reranker_rejects_weak_dense_neighbor_without_lexical_support():
    chunks = [
        {"source":"network.md","content":"Diagnóstico de conectividade de rede e gateway padrão.","score":0.42},
        {"source":"account.md","content":"Falha de acesso a contas de usuário.","score":0.38},
    ]

    ranked = rerank_hybrid("Como configurar WireGuard no Ubuntu?",chunks)

    assert ranked == []


def test_hybrid_reranker_can_rescue_a_strong_semantic_paraphrase():
    chunks = [
        {"source":"phishing.md","content":"Suspeita de phishing por e-mail. Não clique em links nem abra anexos.","score":0.64},
        {"source":"network.md","content":"Diagnóstico de conectividade de rede e gateway.","score":0.41},
    ]

    ranked = rerank_hybrid(
        "Recebi uma mensagem que parece um golpe por email. O que fazer?",
        chunks,
        minimum_lexical_coverage=0.8,
        semantic_rescue_threshold=0.6,
    )

    assert [item["source"] for item in ranked] == ["phishing.md"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semantic_weight":-0.1},
        {"semantic_weight":1.1},
        {"minimum_lexical_coverage":-0.1},
        {"semantic_rescue_threshold":1.1},
    ],
)
def test_hybrid_reranker_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        rerank_hybrid("test query",[],**kwargs)
