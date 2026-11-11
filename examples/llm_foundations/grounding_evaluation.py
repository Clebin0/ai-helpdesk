"""Run the local LLM-as-judge grounding evaluation on a captured answer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .grounding import judge_answer_grounding, summarize_grounding
from .ollama_client import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient, OllamaError

ROOT = Path(__file__).resolve().parents[2]


def load_input(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Não foi possível ler {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("O arquivo de entrada precisa conter um objeto JSON.")
    if not isinstance(payload.get("answer"), str):
        raise ValueError("O arquivo precisa conter answer como string.")
    if not isinstance(payload.get("sources"), list):
        raise ValueError("O arquivo precisa conter sources como lista.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia afirmações de uma resposta contra fontes com Ollama local."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "evaluation" / "grounding.example.json",
        help="JSON com answer e sources.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", DEFAULT_HOST),
        help="Host HTTP do Ollama local.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        help="Modelo local usado como avaliador.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = load_input(args.input.resolve())
        client = OllamaClient(host=args.host, model=args.model)
        assessment = judge_answer_grounding(
            payload["answer"],
            payload["sources"],
            client=client,
        )
        report = summarize_grounding(
            payload["answer"],
            payload["sources"],
            assessment,
        )
    except (OSError, ValueError, OllamaError, ValidationError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
