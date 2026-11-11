"""In-memory semantic retrieval with source-aware citations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .embeddings import (
    MAX_BATCH_SIZE,
    EmbeddingError,
    TextEmbedder,
    validate_vectors,
)

MAX_QUERY_CHARS = 4_000
MAX_TOP_K = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80


@dataclass(frozen=True)
class KnowledgeDocument:
    """A text source the caller is authorized to index."""

    source: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source não pode estar vazio.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content não pode estar vazio.")
        if len(self.content) > 80_000:
            raise ValueError("Um documento não pode exceder 80.000 caracteres.")


@dataclass(frozen=True)
class TextChunk:
    source: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class SearchResult:
    citation: str
    source: str
    chunk_index: int
    content: str
    score: float


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 50
    ):
        raise ValueError("chunk_size deve ser um inteiro de pelo menos 50.")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError("overlap deve ser um inteiro entre 0 e chunk_size - 1.")

    normalized = text.replace("\x00", "").replace("\r\n", "\n").strip()
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Vetores vazios ou com dimensões incompatíveis.")

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not all(math.isfinite(value) for value in left_values + right_values):
        raise ValueError("Vetores devem conter apenas valores finitos.")

    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class SemanticRetriever:
    """Build a small in-memory index and retrieve the closest text chunks.

    This class retrieves evidence only. It does not call an LLM, generate
    answers, execute tools, or infer facts that are absent from the sources.
    """

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        embedder: TextEmbedder,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if (
            isinstance(documents, (str, bytes))
            or not isinstance(documents, Sequence)
            or not documents
        ):
            raise ValueError("Informe uma sequência com pelo menos um documento.")
        if not all(isinstance(document, KnowledgeDocument) for document in documents):
            raise ValueError("Todos os itens precisam ser KnowledgeDocument.")
        sources = [document.source.strip() for document in documents]
        if len(set(sources)) != len(sources):
            raise ValueError("Cada documento precisa ter um source único.")

        self._embedder = embedder
        chunks: list[TextChunk] = []
        for document in documents:
            pieces = chunk_text(
                document.content,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            chunks.extend(
                TextChunk(
                    source=document.source.strip(),
                    chunk_index=index,
                    content=piece,
                )
                for index, piece in enumerate(pieces)
            )

        if not chunks:
            raise ValueError("Não há trechos de texto para indexar.")

        vectors: list[list[float]] = []
        for offset in range(0, len(chunks), MAX_BATCH_SIZE):
            batch_texts = [
                chunk.content
                for chunk in chunks[offset : offset + MAX_BATCH_SIZE]
            ]
            batch_vectors = embedder.embed(batch_texts)
            vectors.extend(
                validate_vectors(batch_vectors, expected_count=len(batch_texts))
            )
        self._vectors = validate_vectors(vectors, expected_count=len(chunks))
        self._chunks = tuple(chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query não pode estar vazia.")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query não pode exceder {MAX_QUERY_CHARS} caracteres.")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= MAX_TOP_K
        ):
            raise ValueError(f"top_k deve ser um inteiro entre 1 e {MAX_TOP_K}.")
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score)
            or not -1 <= min_score <= 1
        ):
            raise ValueError("min_score deve ser um número entre -1 e 1.")

        query_vectors = self._embedder.embed([query.strip()])
        query_vector = validate_vectors(query_vectors, expected_count=1)[0]

        ranked = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if len(query_vector) != len(vector):
                raise EmbeddingError(
                    "O modelo de consulta retornou dimensão diferente da indexação. "
                    "Use o mesmo modelo para indexar e consultar."
                )
            score = cosine_similarity(query_vector, vector)
            if score >= min_score:
                ranked.append((score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                citation=f"S{index}",
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=round(score, 4),
            )
            for index, (score, chunk) in enumerate(ranked[:top_k], start=1)
        ]
