"""Run a small local semantic-search demo over synthetic IT runbooks."""

from __future__ import annotations

import argparse
import os
import sys

from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingError, FastEmbedEncoder
from .semantic_search import KnowledgeDocument, SemanticRetriever


SAMPLE_DOCUMENTS = [
    KnowledgeDocument(
        source="network-dns-runbook.md",
        content=(
            "Diagnóstico de DNS: confirme quantas pessoas e unidades foram afetadas. "
            "No Windows, use ipconfig /all para conferir a configuração de rede e "
            "nslookup para testar a resolução de nomes. Registre o servidor DNS "
            "consultado, o horário e a mensagem retornada. Não altere configurações "
            "de produção sem autorização."
        ),
    ),
    KnowledgeDocument(
        source="printing-runbook.md",
        content=(
            "Diagnóstico de impressão: confirme se a impressora está ligada e conectada. "
            "Confira a fila de impressão, o equipamento padrão e o servidor de impressão. "
            "Registre o modelo, o erro exibido e se outros usuários também foram afetados."
        ),
    ),
    KnowledgeDocument(
        source="account-access-runbook.md",
        content=(
            "Problemas de acesso: confirme a identidade pelo procedimento oficial, "
            "verifique a situação da conta e a permissão solicitada. Nunca peça ou "
            "compartilhe senhas, tokens ou códigos MFA. Encaminhe aprovações ao "
            "responsável pelo sistema."
        ),
    ),
    KnowledgeDocument(
        source="security-incident-runbook.md",
        content=(
            "Suspeita de phishing: oriente a pessoa a não abrir anexos nem informar "
            "credenciais. Registre remetente, horário, URL e indicadores disponíveis. "
            "Acione o fluxo de resposta a incidentes de segurança e preserve as evidências."
        ),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Busca semanticamente em runbooks sintéticos usando embeddings locais."
    )
    parser.add_argument("query", help="Pergunta ou descrição do problema.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument(
        "--model",
        default=os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    args = parser.parse_args()

    try:
        encoder = FastEmbedEncoder(model_name=args.model)
        retriever = SemanticRetriever(SAMPLE_DOCUMENTS, encoder)
        results = retriever.search(
            args.query,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except (EmbeddingError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(
            "Nenhum trecho atingiu o limite mínimo de similaridade. "
            "Não há fonte suficiente para responder com segurança."
        )
        return 0

    print(f"Trechos indexados: {retriever.chunk_count}\n")
    for result in results:
        print(
            f"[{result.citation}] similaridade={result.score:.4f} "
            f"fonte={result.source} trecho={result.chunk_index + 1}"
        )
        print(result.content)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
