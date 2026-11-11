from __future__ import annotations

import math

import pytest

from examples.llm_foundations.embeddings import (
    EmbeddingError,
    FastEmbedEncoder,
    validate_vectors,
)
from examples.llm_foundations.semantic_search import (
    KnowledgeDocument,
    SemanticRetriever,
    chunk_text,
    cosine_similarity,
)


class FakeEmbedder:
    """Deterministic fake embeddings; no model download or network access."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float(any(word in normalized for word in ("dns", "resolver", "rede"))),
                    float(any(word in normalized for word in ("impressora", "fila", "printer"))),
                    float(any(word in normalized for word in ("phishing", "ameaça", "segurança"))),
                ]
            )
        return vectors


def test_chunk_text_preserves_content_and_creates_overlap():
    source = ("Parágrafo de diagnóstico de DNS.\n" * 12).strip()
    chunks = chunk_text(source, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert any(
        set(chunks[index]) & set(chunks[index + 1])
        for index in range(len(chunks) - 1)
    )


def test_chunk_text_rejects_invalid_limits():
    with pytest.raises(ValueError):
        chunk_text("x" * 200, chunk_size=20)
    with pytest.raises(ValueError):
        chunk_text("x" * 200, chunk_size=100, overlap=100)


def test_cosine_similarity_handles_direction_and_zero_vectors():
    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_cosine_similarity_rejects_invalid_vectors():
    with pytest.raises(ValueError):
        cosine_similarity([], [])
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1])
    with pytest.raises(ValueError):
        cosine_similarity([float("nan")], [1])


def test_validate_vectors_checks_count_dimensions_and_finite_values():
    assert validate_vectors([[1, 2], [3, 4]], expected_count=2) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    with pytest.raises(EmbeddingError):
        validate_vectors([[1, 2]], expected_count=2)
    with pytest.raises(EmbeddingError):
        validate_vectors([[1, 2], [3]], expected_count=2)
    with pytest.raises(EmbeddingError):
        validate_vectors([[float("inf")]], expected_count=1)


def test_fastembed_encoder_validates_configuration_without_loading_model():
    encoder = FastEmbedEncoder()
    assert encoder.model_name == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    with pytest.raises(ValueError):
        FastEmbedEncoder(batch_size=0)
    with pytest.raises(ValueError):
        FastEmbedEncoder(model_name="")


def test_retriever_ranks_the_most_relevant_source_first():
    documents = [
        KnowledgeDocument(
            source="network.md",
            content="Procedimento para investigar DNS e servidor resolvedor.",
        ),
        KnowledgeDocument(
            source="printer.md",
            content="Procedimento para fila e impressora sem resposta.",
        ),
        KnowledgeDocument(
            source="security.md",
            content="Procedimento para phishing e incidente de segurança.",
        ),
    ]
    retriever = SemanticRetriever(documents, FakeEmbedder())

    results = retriever.search("Falha no DNS da rede", top_k=3)

    assert retriever.chunk_count == 3
    assert results[0].source == "network.md"
    assert results[0].citation == "S1"
    assert results[0].score == 1.0


def test_retriever_applies_minimum_similarity_and_returns_no_guesses():
    documents = [
        KnowledgeDocument("printer.md", "Fila da impressora travada."),
        KnowledgeDocument("network.md", "DNS e servidor resolvedor."),
    ]
    retriever = SemanticRetriever(documents, FakeEmbedder())

    results = retriever.search("Falha de autenticação desconhecida", min_score=0.5)

    assert results == []


def test_retriever_checks_sources_and_embedding_dimensions():
    with pytest.raises(ValueError):
        SemanticRetriever(
            [
                KnowledgeDocument("same.md", "Primeiro texto suficientemente longo."),
                KnowledgeDocument("same.md", "Segundo texto suficientemente longo."),
            ],
            FakeEmbedder(),
        )

    class DifferentQueryDimension:
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                return [[1.0, 0.0, 0.0] for _ in texts]
            return [[1.0, 0.0] for _ in texts]

    retriever = SemanticRetriever(
        [KnowledgeDocument("network.md", "DNS e servidor resolvedor.")],
        DifferentQueryDimension(),
    )
    with pytest.raises(EmbeddingError, match="dimensão diferente"):
        retriever.search("DNS falha")


@pytest.mark.parametrize("query", ["", "   ", "x" * 4001])
def test_retriever_validates_query(query):
    retriever = SemanticRetriever(
        [KnowledgeDocument("network.md", "DNS e servidor resolvedor.")],
        FakeEmbedder(),
    )
    with pytest.raises(ValueError):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, 21, True, 1.5])
def test_retriever_validates_top_k(top_k):
    retriever = SemanticRetriever(
        [KnowledgeDocument("network.md", "DNS e servidor resolvedor.")],
        FakeEmbedder(),
    )
    with pytest.raises(ValueError):
        retriever.search("DNS", top_k=top_k)


@pytest.mark.parametrize("min_score", [-1.1, 1.1, float("nan"), True])
def test_retriever_validates_min_score(min_score):
    retriever = SemanticRetriever(
        [KnowledgeDocument("network.md", "DNS e servidor resolvedor.")],
        FakeEmbedder(),
    )
    with pytest.raises(ValueError):
        retriever.search("DNS", min_score=min_score)



def test_retriever_batches_large_indexes():
    class CountingEmbedder:
        def __init__(self):
            self.batch_sizes = []

        def embed(self, texts):
            self.batch_sizes.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    embedder = CountingEmbedder()
    documents = [
        KnowledgeDocument(
            source=f"document-{index}.md",
            content=f"Documento sintético de suporte número {index}.",
        )
        for index in range(130)
    ]

    retriever = SemanticRetriever(documents, embedder)
    results = retriever.search("Documento de suporte")

    assert retriever.chunk_count == 130
    assert embedder.batch_sizes == [128, 2, 1]
    assert len(results) == 4


def test_retriever_rejects_non_document_items():
    with pytest.raises(ValueError, match="KnowledgeDocument"):
        SemanticRetriever(["texto sem metadados"], FakeEmbedder())
