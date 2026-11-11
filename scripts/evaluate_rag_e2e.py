"""Run a local end-to-end RAG benchmark with answer-grounding checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_e2e_evaluation import evaluate_end_to_end, select_balanced_cases


NO_CONTEXT_ANSWER = (
    "Não encontrei trechos suficientemente relacionados na base de conhecimento. "
    "Não vou completar a resposta com suposições. Adicione documentação relevante "
    "ou reformule a pergunta."
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Não foi possível ler {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"O arquivo {path} precisa conter um objeto JSON.")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_json(args.dataset.resolve())
    documents = dataset.get("documents", [])
    cases = dataset.get("queries", [])
    if not isinstance(documents, list) or not documents:
        raise SystemExit("O benchmark precisa conter documents.")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("O benchmark precisa conter queries.")
    selected_cases = select_balanced_cases(cases, args.limit)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    # This experiment is deliberately local-only: do not spend hosted API credits.
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["RAG_EMBEDDING_PROVIDER"] = "local"
    os.environ["RAG_DB_PATH"] = str(args.db_path.resolve())

    ollama_base = (
        args.ollama_host
        or os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://127.0.0.1:11434"
    ).rstrip("/")
    if ollama_base.endswith("/v1"):
        ollama_base = ollama_base[:-3]
    model = args.model or os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen3.5:4b-q4_K_M"
    os.environ["OLLAMA_HOST"] = ollama_base
    os.environ["OLLAMA_BASE_URL"] = ollama_base
    os.environ["OLLAMA_MODEL"] = model
    os.environ["OLLAMA_CHAT_MODEL"] = model

    try:
        import assistant_core
        from examples.llm_foundations.grounding import (
            judge_answer_grounding,
            summarize_grounding,
        )
        from examples.llm_foundations.ollama_client import OllamaClient, OllamaError

        index_started = time.perf_counter()
        for document in documents:
            assistant_core.ingest_document(
                str(document["source"]),
                str(document["content"]),
            )
        indexing_seconds = time.perf_counter() - index_started

        client = OllamaClient(host=ollama_base, model=model, timeout=args.timeout)
        executions: list[dict[str, Any]] = []
        generation_calls = 0
        judge_calls = 0
        for index, case in enumerate(selected_cases, start=1):
            question = str(case["question"])
            retrieved = assistant_core.relevant_chunks(question, top_k=args.top_k)
            sources = assistant_core.source_metadata(retrieved)
            grounding_report = None
            used_llm = bool(retrieved)

            if retrieved:
                answer = assistant_core.generate_answer(question, retrieved)
                generation_calls += 1
                evidence = [
                    {
                        "citation": f"S{source_index}",
                        "source": str(item["source"]),
                        "content": str(item["content"]),
                    }
                    for source_index, item in enumerate(retrieved, start=1)
                ]
                assessment = judge_answer_grounding(
                    answer,
                    evidence,
                    client=client,
                )
                judge_calls += 1
                grounding_report = summarize_grounding(
                    answer,
                    evidence,
                    assessment,
                )
            else:
                answer = NO_CONTEXT_ANSWER

            executions.append(
                {
                    "query_id": str(case["id"]),
                    "answer": answer,
                    "retrieved_sources": sources,
                    "used_llm": used_llm,
                    "grounding": grounding_report,
                }
            )
            print(
                f"[{index}/{len(selected_cases)}] {case['id']}: "
                f"{len(retrieved)} trecho(s), "
                f"{'com geração local' if used_llm else 'abstenção sem contexto'}",
                file=sys.stderr,
            )

        report = evaluate_end_to_end(selected_cases, executions)
        report.update(
            {
                "benchmark": dataset.get("name", "RAG end-to-end benchmark"),
                "dataset_version": dataset.get("version", 1),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chat_provider": "ollama_local",
                "chat_model": model,
                "embedding_provider": "fastembed_local",
                "embedding_model": assistant_core.embedding_model(),
                "top_k": args.top_k,
                "retrieval_min_relevance": float(os.getenv("RAG_MIN_RELEVANCE", "0.18")),
                "indexing_seconds": round(indexing_seconds, 3),
                "generation_calls": generation_calls,
                "grounding_judge_calls": judge_calls,
                "selected_query_ids": [str(case["id"]) for case in selected_cases],
                "database_path": str(args.db_path.resolve()),
            }
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Falha na avaliação end-to-end: {exc}") from exc
    except Exception as exc:
        if exc.__class__.__name__ in {"OllamaError", "ValidationError"}:
            raise SystemExit(f"Falha na avaliação end-to-end: {exc}") from exc
        raise

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
        description=(
            "Avalia recuperação, resposta gerada, citações, fundamentação e abstenção "
            "usando Ollama e FastEmbed locais."
        )
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
        default=ROOT / "data" / "rag-e2e-benchmark.sqlite3",
        help="SQLite isolado; não modifica a base do assistente.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Trechos recuperados por pergunta (1 a 8).")
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Amostra balanceada por tipo de pergunta; use 21 para o benchmark completo.",
    )
    parser.add_argument("--ollama-host", help="URL local do Ollama; padrão 127.0.0.1:11434.")
    parser.add_argument("--model", help="Modelo Ollama local; padrão qwen3.5:4b-q4_K_M.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout de cada avaliação do juiz.")
    parser.add_argument("--output", type=Path, help="Salvar relatório JSON.")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 8:
        parser.error("--top-k precisa ficar entre 1 e 8.")
    if args.limit < 2:
        parser.error("--limit precisa ser pelo menos 2 para incluir ambos os tipos de pergunta.")
    if args.timeout <= 0:
        parser.error("--timeout precisa ser maior que zero.")
    return args


if __name__ == "__main__":
    run(parse_args())
