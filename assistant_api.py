from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import assistant_core

router = APIRouter(prefix="/assistant", tags=["LLM RAG Assistant"])


class AssistantInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class IngestRequest(AssistantInput):
    source: str = Field(min_length=2, max_length=160)
    content: str = Field(min_length=20, max_length=80_000)


class ChatTurn(AssistantInput):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1_200)


class ChatRequest(AssistantInput):
    question: str = Field(min_length=5, max_length=3_000)
    top_k: int = Field(default=4, ge=1, le=8)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


def _authorize(provided: str | None, variable: str) -> None:
    expected = os.getenv(variable)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"{variable} is not configured. See docs/LLM_RAG.md.",
        )
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing access token.")


def _provider_rate_limit_error(exc: Exception) -> HTTPException | None:
    status = getattr(exc, "status_code", None)
    error_code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    response = getattr(exc, "response", None)

    if response is not None:
        try:
            payload = response.json()
            provider_error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(provider_error, dict):
                error_code = error_code or provider_error.get("code")
                error_type = error_type or provider_error.get("type")
        except Exception:
            pass

    quota_codes = {
        "credit_balance_exhausted",
        "insufficient_quota",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
    if status == 402 or error_code in quota_codes or error_type == "insufficient_quota":
        return HTTPException(
            status_code=402,
            detail=(
                "A cota de inferência da Hugging Face foi esgotada ou a conta não tem "
                "créditos disponíveis. O aplicativo não compra créditos nem habilita "
                "cobrança adicional."
            ),
        )

    if status == 429:
        return HTTPException(
            status_code=429,
            detail=(
                "O provedor limitou temporariamente as solicitações. Aguarde e tente "
                "novamente; evite repetir chamadas em sequência."
            ),
        )

    if status in {401, 403}:
        return HTTPException(
            status_code=503,
            detail=(
                "A autenticação da Hugging Face falhou ou o token não tem permissão "
                "Inference Providers. Revise HF_TOKEN e as permissões do token."
            ),
        )

    return None


def _ndjson(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _chat_error(exc: Exception) -> str:
    provider_error = _provider_rate_limit_error(exc)
    if provider_error is not None:
        return str(provider_error.detail)
    if isinstance(exc, RuntimeError):
        return str(exc)
    return "Não foi possível gerar a resposta. Verifique se o Ollama está ativo e tente novamente."


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def assistant_chat_ui():
    page = Path(__file__).resolve().with_name("assistant_chat.html")
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Chat interface not found.")
    return page.read_text(encoding="utf-8")


@router.get("/health")
def assistant_health():
    return {
        "status": "ready",
        "llm_provider": assistant_core.llm_provider(),
        "chat_model": assistant_core.chat_model(),
        "hf_token_configured": bool(
            os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        ),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "embedding_provider": assistant_core.embedding_provider(),
        "embedding_model": assistant_core.embedding_model(),
        "chat_access_configured": bool(os.getenv("ASSISTANT_ACCESS_TOKEN")),
        "ingestion_access_configured": bool(os.getenv("ASSISTANT_ADMIN_TOKEN")),
    }


@router.get("/documents")
def assistant_documents(x_admin_token: str | None = Header(default=None)):
    _authorize(x_admin_token, "ASSISTANT_ADMIN_TOKEN")
    return {"documents": assistant_core.list_documents()}


@router.post("/ingest")
def assistant_ingest(
    body: IngestRequest,
    x_admin_token: str | None = Header(default=None),
):
    _authorize(x_admin_token, "ASSISTANT_ADMIN_TOKEN")
    try:
        result = assistant_core.ingest_document(body.source, body.content)
        return {"status": "indexed", **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        provider_error = _provider_rate_limit_error(exc)
        if provider_error is not None:
            raise provider_error from exc
        raise HTTPException(
            status_code=502,
            detail="Document indexing failed. Check server logs and provider configuration.",
        ) from exc


@router.post("/chat")
def assistant_chat(
    body: ChatRequest,
    x_assistant_token: str | None = Header(default=None),
):
    _authorize(x_assistant_token, "ASSISTANT_ACCESS_TOKEN")
    history = [turn.model_dump() for turn in body.history]
    try:
        return assistant_core.answer_question(
            body.question,
            top_k=body.top_k,
            history=history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        provider_error = _provider_rate_limit_error(exc)
        if provider_error is not None:
            raise provider_error from exc
        raise HTTPException(
            status_code=502,
            detail="Assistant request failed. Check server logs and provider configuration.",
        ) from exc


@router.post("/chat/stream")
def assistant_chat_stream(
    body: ChatRequest,
    x_assistant_token: str | None = Header(default=None),
):
    _authorize(x_assistant_token, "ASSISTANT_ACCESS_TOKEN")
    history = [turn.model_dump() for turn in body.history]

    try:
        query = assistant_core.retrieval_query(body.question, history)
        relevant = assistant_core.relevant_chunks(query, top_k=body.top_k)
        sources = assistant_core.source_metadata(relevant)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        provider_error = _provider_rate_limit_error(exc)
        if provider_error is not None:
            raise provider_error from exc
        raise HTTPException(
            status_code=502,
            detail="Assistant retrieval failed. Check server logs and local embeddings.",
        ) from exc

    def events():
        yield _ndjson(
            {
                "type": "meta",
                "model": assistant_core.chat_model(),
                "retrieval_model": assistant_core.embedding_model(),
                "sources": sources,
                "used_llm": bool(relevant),
            }
        )

        if not relevant:
            yield _ndjson(
                {
                    "type": "token",
                    "text": (
                        "Não encontrei trechos suficientemente relacionados na base de "
                        "conhecimento. Não vou completar a resposta com suposições. "
                        "Adicione documentação relevante ou reformule a pergunta."
                    ),
                }
            )
            yield _ndjson({"type": "done"})
            return

        try:
            for fragment in assistant_core.generate_answer_stream(
                body.question, relevant, history
            ):
                if fragment:
                    yield _ndjson({"type": "token", "text": fragment})
            yield _ndjson({"type": "done"})
        except Exception as exc:
            yield _ndjson({"type": "error", "detail": _chat_error(exc)})

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
