from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "as", "ao", "aos", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "essa", "esse", "esta", "este", "eu", "foi", "for",
    "is", "isso", "isto", "ja", "lhe", "mais", "mas", "me", "meu", "minha",
    "na", "nas", "no", "nos", "o", "os", "ou", "para", "pela", "pelo", "por",
    "qual", "quando", "que", "quem", "se", "sem", "ser", "seu", "sua", "tambem",
    "te", "tem", "ter", "um", "uma", "voce", "your",
}


def tokenize(text: str) -> list[str]:
    """Create accent-insensitive tokens and normalize common technical forms."""
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("e-mail", "email").replace("e mail", "email")
    normalized = normalized.replace("wi-fi", "wifi").replace("wi fi", "wifi")
    # Small domain-safe normalization for common Portuguese/English plurals.
    token_forms = {
        "links": "link",
        "sites": "site",
        "anexos": "anexo",
        "arquivos": "arquivo",
        "usuarios": "usuario",
        "redes": "rede",
        "processos": "processo",
        "documentos": "documento",
        "politicas": "politica",
        "chaves": "chave",
        "suspeito": "suspeita",
        "suspeitos": "suspeita",
        "trazer": "restaurar",
        "volta": "restaurar",
        "recuperar": "restaurar",
        "restauracao": "restaurar",
    }
    return [
        token_forms.get(token, token)
        for token in _TOKEN_PATTERN.findall(normalized)
        if len(token) > 1 and token not in _STOP_WORDS
    ]


def rerank_hybrid(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    semantic_weight: float = 0.55,
    minimum_lexical_coverage: float = 0.25,
    semantic_rescue_threshold: float = 0.60,
) -> list[dict[str, Any]]:
    """Rerank dense candidates using IDF-weighted lexical query coverage.

    Experimental, dependency-free baseline. It is not the production retriever.
    A strong semantic score can rescue paraphrases with limited token overlap.
    """
    if not 0 <= semantic_weight <= 1:
        raise ValueError("semantic_weight must be between 0 and 1.")
    if not 0 <= minimum_lexical_coverage <= 1:
        raise ValueError("minimum_lexical_coverage must be between 0 and 1.")
    if not 0 <= semantic_rescue_threshold <= 1:
        raise ValueError("semantic_rescue_threshold must be between 0 and 1.")
    if not chunks:
        return []

    query_terms = set(tokenize(question))
    if not query_terms:
        return []

    tokenized_documents = [
        tokenize(str(chunk.get("content", ""))) for chunk in chunks
    ]
    document_sets = [set(tokens) for tokens in tokenized_documents]
    document_count = len(chunks)
    document_frequency = {
        term: sum(term in terms for terms in document_sets)
        for term in query_terms
    }
    idf = {
        term: math.log(
            1 + (document_count - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in document_frequency.items()
    }
    total_query_weight = sum(idf.values()) or 1.0

    ranked: list[dict[str, Any]] = []
    for chunk, terms in zip(chunks, document_sets):
        semantic_score = max(0.0, min(1.0, float(chunk.get("score", 0.0))))
        lexical_coverage = (
            sum(idf[term] for term in query_terms if term in terms)
            / total_query_weight
        )

        # Avoid accepting every dense nearest-neighbour by default. Retain a
        # semantic rescue path for strong paraphrase matches.
        if (
            lexical_coverage < minimum_lexical_coverage
            and semantic_score < semantic_rescue_threshold
        ):
            continue

        hybrid_score = (
            semantic_weight * semantic_score
            + (1 - semantic_weight) * lexical_coverage
        )
        item = dict(chunk)
        item["semantic_score"] = round(semantic_score, 4)
        item["lexical_coverage"] = round(lexical_coverage, 4)
        item["score"] = round(hybrid_score, 4)
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked
