"""Local text embeddings backed by FastEmbed.

The model is loaded lazily: importing this module does not download or initialize
model files. Tests can inject a small deterministic embedder instead.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MAX_BATCH_SIZE = 128
MAX_TEXT_CHARS = 10_000


class EmbeddingError(RuntimeError):
    """Raised when a local embedding model cannot encode text safely."""


class TextEmbedder(Protocol):
    """Minimal interface required by the semantic retriever."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a non-empty sequence of strings into numeric vectors."""


def validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise ValueError("texts deve ser uma sequência de strings.")
    if not texts or len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"Envie entre 1 e {MAX_BATCH_SIZE} textos.")

    validated: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"O texto {index + 1} está vazio ou é inválido.")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(
                f"O texto {index + 1} excede {MAX_TEXT_CHARS} caracteres."
            )
        validated.append(text.strip())
    return validated


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingError(
            "A quantidade de embeddings retornados não corresponde aos textos."
        )

    validated: list[list[float]] = []
    dimension: int | None = None
    for index, vector in enumerate(vectors):
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                f"O embedding {index + 1} contém valores inválidos."
            ) from exc

        if not values:
            raise EmbeddingError(f"O embedding {index + 1} está vazio.")
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingError(
                f"O embedding {index + 1} contém valores não finitos."
            )
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise EmbeddingError("Os embeddings possuem dimensões diferentes.")
        validated.append(values)
    return validated


@dataclass
class FastEmbedEncoder:
    """Lazy FastEmbed wrapper for multilingual local embeddings."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    _model: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name não pode estar vazio.")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= MAX_BATCH_SIZE
        ):
            raise ValueError("batch_size deve ser um inteiro entre 1 e 128.")

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding
                    except ImportError as exc:
                        raise EmbeddingError(
                            "FastEmbed não está instalado. Execute: "
                            "python -m pip install -r requirements.txt"
                        ) from exc
                    try:
                        self._model = TextEmbedding(model_name=self.model_name)
                    except Exception as exc:
                        raise EmbeddingError(
                            "Não foi possível carregar o modelo de embeddings. "
                            "Confira o nome do modelo e a conexão necessária para o primeiro download."
                        ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        validated = validate_texts(texts)
        try:
            raw_vectors = list(
                self._get_model().embed(
                    validated,
                    batch_size=self.batch_size,
                )
            )
        except Exception as exc:
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError("Falha ao gerar embeddings localmente.") from exc
        return validate_vectors(raw_vectors, expected_count=len(validated))
