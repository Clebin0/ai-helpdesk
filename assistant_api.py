from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import assistant_core

router = APIRouter(prefix="/assistant", tags=["LLM RAG Assistant"])


class AssistantInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class IngestRequest(AssistantInput):
    source: str = Field(min_length=2, max_length=160)
    content: str = Field(min_length=20, max_length=80_000)


class ChatRequest(AssistantInput):
    question: str = Field(min_length=5, max_length=3_000)
    top_k: int = Field(default=4, ge=1, le=8)


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
                "créditos disponíveis. O plano gratuito inclui uma pequena cota mensal; "
                "o aplicativo não compra créditos nem habilita cobrança adicional."
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
    try:
        return assistant_core.answer_question(body.question, top_k=body.top_k)
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
