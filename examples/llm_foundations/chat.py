"""Run a single local chat completion from the terminal."""

from __future__ import annotations

import argparse
import os
import sys

from .ollama_client import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient, OllamaError


SYSTEM_PROMPT = (
    "Você é um assistente técnico didático. Responda em português brasileiro, "
    "explique conceitos com exemplos curtos e sinalize quando não souber."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Converse com um modelo local servido pelo Ollama."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Pergunta para o modelo. Sem este argumento, o programa pergunta no terminal.",
    )
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", DEFAULT_HOST))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    question = (args.question or input("Você: ")).strip()
    if not question:
        parser.error("A pergunta não pode estar vazia.")

    try:
        client = OllamaClient(host=args.host, model=args.model)
        answer = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=args.temperature,
        )
    except (OllamaError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"\nAssistente ({args.model}):\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
