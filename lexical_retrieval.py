from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from hybrid_retrieval import tokenize


class FTS5Index:
    """Small, isolated SQLite FTS5 index for lexical BM25 retrieval.

    The index stores normalized tokens only. Original chunks are kept in a
    rowid map so callers receive the exact source text used by the RAG system.
    """

    def __init__(
        self,
        chunks: Iterable[dict[str, Any]],
        *,
        db_path: str | Path = ":memory:",
    ) -> None:
        self.connection = sqlite3.connect(str(db_path))
        self._original_chunks: dict[int, dict[str, Any]] = {}
        try:
            self.connection.execute(
                """
                CREATE VIRTUAL TABLE lexical_fts USING fts5(
                    source UNINDEXED,
                    chunk_index UNINDEXED,
                    content,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
            for chunk in chunks:
                source = str(chunk.get("source", "")).strip()
                content = str(chunk.get("content", "")).strip()
                if not source or not content:
                    continue
                normalized_content = " ".join(tokenize(content))
                if not normalized_content:
                    continue
                cursor = self.connection.execute(
                    """
                    INSERT INTO lexical_fts (source, chunk_index, content)
                    VALUES (?, ?, ?)
                    """,
                    (source, int(chunk.get("chunk_index", 0)), normalized_content),
                )
                self._original_chunks[int(cursor.lastrowid)] = dict(chunk)
            self.connection.commit()
        except sqlite3.OperationalError as exc:
            self.connection.close()
            raise RuntimeError(
                "SQLite FTS5 is unavailable in this Python build."
            ) from exc

    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks; lower (more negative) BM25 is better."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        terms = tokenize(query)
        if not terms:
            return []

        # Every token is quoted so user text cannot inject FTS operators.
        match_expression = " OR ".join(
            '"' + term.replace('"', '""') + '"' for term in terms
        )
        rows = self.connection.execute(
            """
            SELECT rowid, bm25(lexical_fts) AS bm25_score
            FROM lexical_fts
            WHERE lexical_fts MATCH ?
            ORDER BY bm25_score ASC, rowid ASC
            LIMIT ?
            """,
            (match_expression, limit),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            item = dict(self._original_chunks[int(row[0])])
            item["bm25_score"] = float(row[1])
            item["lexical_rank"] = rank
            results.append(item)
        return results

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FTS5Index":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def reciprocal_rank_fusion(
    rankings: Iterable[list[dict[str, Any]]],
    *,
    rank_constant: int = 60,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Fuse ranked chunk lists using RRF, deduplicating identical chunks.

    RRF is score-scale agnostic: rank 1 in each input contributes
    1 / (rank_constant + 1). The result preserves original chunk content.
    """
    if (
        not isinstance(rank_constant, int)
        or isinstance(rank_constant, bool)
        or rank_constant < 1
    ):
        raise ValueError("rank_constant must be a positive integer.")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    fused: dict[tuple[str, int], dict[str, Any]] = {}
    for ranking in rankings:
        seen_in_ranking: set[tuple[str, int]] = set()
        for rank, chunk in enumerate(ranking, start=1):
            source = str(chunk.get("source", "")).strip()
            if not source:
                continue
            chunk_index = int(chunk.get("chunk_index", 0))
            key = (source, chunk_index)
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)

            if key not in fused:
                fused[key] = {
                    **chunk,
                    "rrf_score": 0.0,
                    "rrf_ranks": [],
                }
            fused[key]["rrf_score"] += 1.0 / (rank_constant + rank)
            fused[key]["rrf_ranks"].append(rank)

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(item["rrf_ranks"]),
            str(item["source"]).casefold(),
            int(item.get("chunk_index", 0)),
        ),
    )
    results: list[dict[str, Any]] = []
    for item in ordered[:limit]:
        item["rrf_score"] = round(float(item["rrf_score"]), 8)
        item["score"] = item["rrf_score"]
        results.append(item)
    return results
