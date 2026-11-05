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


@router.get("/health")
def assistant_health():
    return {
        "status": "ready",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "chat_access_configured": bool(os.getenv("ASSISTANT_ACCESS_TOKEN")),
        "ingestion_access_configured": bool(os.getenv("ASSISTANT_ADMIN_TOKEN")),
        "chat_model": assistant_core.chat_model(),
        "embedding_model": assistant_core.embedding_model(),
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
        # Avoid returning provider messages, request IDs or secrets to API clients.
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
        raise HTTPException(
            status_code=502,
            detail="Assistant request failed. Check server logs and provider configuration.",
        ) from exc
