from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_retrieval import rerank_hybrid
from rag_evaluation import evaluate_abstention, evaluate_citation_answers, evaluate_retrieval


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[index]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Não foi possível ler {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"O arquivo {path} precisa conter um objeto JSON.")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    dataset = _load_json(dataset_path)
    documents = dataset.get("documents", [])
    cases = dataset.get("queries", [])
    if not documents or not cases:
        raise SystemExit("O benchmark precisa conter documents e queries.")

    expected_names = {str(item.get("source", "")).strip() for item in documents}
    query_ids: set[str] = set()
    for case in cases:
        query_id = str(case.get("id", "")).strip()
        if not query_id or query_id in query_ids:
            raise SystemExit(f"ID de pergunta ausente ou duplicado: {query_id!r}")
        query_ids.add(query_id)
        if not isinstance(case.get("answerable"), bool):
            raise SystemExit(f"A pergunta {query_id} precisa definir answerable como boolean.")
        expected_sources = set(case.get("expected_sources", []))
        unknown = expected_sources - expected_names
        if unknown:
            raise SystemExit(
                f"Fontes esperadas não existem no corpus ({query_id}): {sorted(unknown)}"
            )
        if case["answerable"] and not expected_sources:
            raise SystemExit(f"A pergunta respondível {query_id} precisa de expected_sources.")
        if not case["answerable"] and expected_sources:
            raise SystemExit(f"A pergunta sem resposta {query_id} precisa de expected_sources vazio.")

    db_path = args.db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if args.min_relevance is not None:
        os.environ["RAG_MIN_RELEVANCE"] = str(args.min_relevance)

    # Keep benchmark vectors isolated from the user's assistant knowledge base.
    os.environ["RAG_DB_PATH"] = str(db_path)

    try:
        import assistant_core
    except Exception as exc:
        raise SystemExit(f"Não foi possível importar assistant_core: {exc}") from exc

    index_started = time.perf_counter()
    indexed_documents = []
    for document in documents:
        result = assistant_core.ingest_document(
            str(document["source"]),
            str(document["content"]),
        )
        indexed_documents.append(result)
    indexing_seconds = time.perf_counter() - index_started

    retrieved_by_query: dict[str, list[dict[str, Any]]] = {}
    query_latencies_ms: list[float] = []
    per_query_latency: list[dict[str, Any]] = []
    minimum_score = float(os.getenv("RAG_MIN_RELEVANCE", "0.18"))

    for case in cases:
        samples: list[float] = []
        retrieved: list[dict[str, Any]] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            # Semantic mode reproduces the live assistant. Hybrid mode reranks
            # up to eight dense candidates with lexical query coverage.
            if args.retrieval_mode == "semantic":
                retrieved = assistant_core.relevant_chunks(
                    str(case["question"]),
                    top_k=args.top_k,
                )
            else:
                candidates = assistant_core.retrieve_chunks(
                    str(case["question"]),
                    top_k=8,
                )
                ranked = rerank_hybrid(
                    str(case["question"]),
                    candidates,
                    semantic_weight=args.hybrid_semantic_weight,
                    minimum_lexical_coverage=args.hybrid_min_lexical_coverage,
                    semantic_rescue_threshold=args.hybrid_semantic_rescue,
                )
                retrieved = [
                    item for item in ranked if item["score"] >= minimum_score
                ][: args.top_k]
            samples.append((time.perf_counter() - started) * 1000)

        retrieved_by_query[str(case["id"])] = retrieved
        average_latency = statistics.mean(samples)
        query_latencies_ms.extend(samples)
        per_query_latency.append(
            {
                "query_id": case["id"],
                "mean_ms": round(average_latency, 3),
                "samples_ms": [round(value, 3) for value in samples],
            }
        )

    answerable_cases = [case for case in cases if case["answerable"]]
    retrieval = evaluate_retrieval(
        answerable_cases,
        retrieved_by_query,
        top_k=args.top_k,
    )
    answerability = evaluate_abstention(
        cases,
        retrieved_by_query,
        top_k=args.top_k,
    )
    retrieval_by_type: dict[str, Any] = {}
    for query_type in sorted({str(case.get("type", "canonical")) for case in answerable_cases}):
        typed_cases = [
            case for case in answerable_cases
            if str(case.get("type", "canonical")) == query_type
        ]
        retrieval_by_type[query_type] = evaluate_retrieval(
            typed_cases,
            retrieved_by_query,
            top_k=args.top_k,
        )
    report: dict[str, Any] = {
        "benchmark": dataset.get("name", "RAG benchmark"),
        "dataset_version": dataset.get("version", 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": assistant_core.embedding_model(),
        "chat_model": assistant_core.chat_model(),
        "llm_calls": 0,
        "top_k": args.top_k,
        "repeats": args.repeats,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_parameters": (
            {
                "semantic_weight": args.hybrid_semantic_weight,
                "lexical_weight": round(1 - args.hybrid_semantic_weight, 4),
                "minimum_lexical_coverage": args.hybrid_min_lexical_coverage,
                "semantic_rescue_threshold": args.hybrid_semantic_rescue,
            }
            if args.retrieval_mode == "hybrid" else None
        ),
        "retrieval_min_relevance": float(os.getenv("RAG_MIN_RELEVANCE", "0.18")),
        "document_count": len(documents),
        "query_count": len(cases),
        "database_path": str(db_path),
        "indexing_seconds": round(indexing_seconds, 3),
        "latency_ms": {
            "mean": round(statistics.mean(query_latencies_ms), 3),
            "median": round(statistics.median(query_latencies_ms), 3),
            "p95": round(_percentile(query_latencies_ms, 95), 3),
            "samples": len(query_latencies_ms),
            "per_query": per_query_latency,
        },
        "retrieval": retrieval,
        "retrieval_by_type": retrieval_by_type,
        "answerability": answerability,
        "notes": [
            "A avaliação de recuperação mede fontes esperadas apenas em perguntas respondíveis.",
            "A abstenção é medida no retriever: zero trechos relevantes significa que o assistente segue o caminho sem contexto; o texto de recusa do LLM não é avaliado.",
            "O benchmark usa documentos e perguntas sintéticos, não dados de clientes ou da empresa.",
            "A latência mede consulta e embedding da pergunta após a indexação; não inclui download inicial do modelo.",
            "O benchmark usa um SQLite separado e não modifica a base de conhecimento padrão do assistente.",
        ],
    }

    if args.answers:
        answers_data = json.loads(args.answers.resolve().read_text(encoding="utf-8"))
        if not isinstance(answers_data, list):
            raise SystemExit("O arquivo --answers precisa conter uma lista JSON.")
        report["citation_audit"] = evaluate_citation_answers(answers_data, cases)

    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        report["report_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local de recuperação RAG para o AI Helpdesk."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evaluation" / "benchmark.json",
        help="Corpus e perguntas sintéticas em JSON.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "data" / "rag-benchmark.sqlite3",
        help="SQLite isolado para o benchmark.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Número de fontes únicas avaliadas (1 a 8).")
    parser.add_argument("--repeats", type=int, default=3, help="Repetições por pergunta para medir latência.")
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=None,
        help="Sobrescreve RAG_MIN_RELEVANCE apenas durante esta execução (0 a 1).",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("semantic", "hybrid"),
        default="semantic",
        help="semantic reproduz o retriever atual; hybrid ativa o reranker lexical experimental.",
    )
    parser.add_argument(
        "--hybrid-semantic-weight",
        type=float,
        default=0.55,
        help="Peso do score vetorial no reranking híbrido (0 a 1).",
    )
    parser.add_argument(
        "--hybrid-min-lexical-coverage",
        type=float,
        default=0.25,
        help="Cobertura lexical IDF mínima para aceitar um candidato (0 a 1).",
    )
    parser.add_argument(
        "--hybrid-semantic-rescue",
        type=float,
        default=0.60,
        help="Score vetorial mínimo para aceitar paráfrase com baixa cobertura lexical (0 a 1).",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        help="JSON opcional com respostas reais e metadados de fontes para auditar citações.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Caminho opcional para salvar o relatório JSON.",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 8:
        parser.error("--top-k precisa ficar entre 1 e 8.")
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats precisa ficar entre 1 e 20.")
    if args.min_relevance is not None and not 0 <= args.min_relevance <= 1:
        parser.error("--min-relevance precisa ficar entre 0 e 1.")
    if not 0 <= args.hybrid_semantic_weight <= 1:
        parser.error("--hybrid-semantic-weight precisa ficar entre 0 e 1.")
    if not 0 <= args.hybrid_min_lexical_coverage <= 1:
        parser.error("--hybrid-min-lexical-coverage precisa ficar entre 0 e 1.")
    if not 0 <= args.hybrid_semantic_rescue <= 1:
        parser.error("--hybrid-semantic-rescue precisa ficar entre 0 e 1.")
    return args


if __name__ == "__main__":
    run(parse_args())
