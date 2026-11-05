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

from rag_evaluation import evaluate_citation_answers, evaluate_retrieval


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
    for case in cases:
        unknown = set(case.get("expected_sources", [])) - expected_names
        if unknown:
            raise SystemExit(
                f"Fontes esperadas não existem no corpus ({case.get('id')}): {sorted(unknown)}"
            )

    db_path = args.db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

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
    candidate_limit = min(8, max(args.top_k, args.top_k * 2))

    for case in cases:
        samples: list[float] = []
        retrieved: list[dict[str, Any]] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            # Evaluate the same relevance threshold used by the live assistant,
            # not the raw unfiltered nearest-neighbour list.
            retrieved = assistant_core.relevant_chunks(
                str(case["question"]),
                top_k=candidate_limit,
            )
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

    retrieval = evaluate_retrieval(
        cases,
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
        "notes": [
            "A avaliação mede recuperação de fontes, não qualidade semântica das respostas do LLM.",
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
    return args


if __name__ == "__main__":
    run(parse_args())
