from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from ml import load_model, read_comparison, read_confusion, read_metrics

LOW_CONFIDENCE_THRESHOLD = 0.72
LOW_MARGIN_THRESHOLD = 0.12

app = FastAPI(title="AI Helpdesk ML", version="4.1.0")
MODEL = load_model()
MODEL_METRICS = read_metrics()

PLAYBOOKS = {
    "network": [
        "Confirmar escopo e usuarios afetados",
        "Validar link, IP e gateway",
        "Testar DNS e destino",
        "Registrar evidencias antes de alterar configuracoes",
    ],
    "hardware": [
        "Registrar modelo e sintomas",
        "Verificar recursos e eventos do sistema",
        "Executar testes nao destrutivos",
        "Planejar troca somente apos diagnostico",
    ],
    "software": [
        "Registrar mensagem de erro",
        "Reproduzir o problema",
        "Validar versao, logs e dependencias",
        "Testar correcao em escopo controlado",
    ],
    "access": [
        "Confirmar identidade pelo processo oficial",
        "Validar conta, bloqueio e permissao",
        "Evitar compartilhar credenciais",
        "Escalar privilegios ao responsavel",
    ],
    "printing": [
        "Validar energia e conectividade",
        "Verificar fila e impressora padrao",
        "Testar pagina de diagnostico",
        "Registrar erro do driver ou spooler",
    ],
    "security": [
        "Isolar o risco sem apagar evidencias",
        "Orientar o usuario a nao interagir mais",
        "Registrar origem, horario e indicadores",
        "Escalar ao processo de seguranca",
    ],
    "other": [
        "Coletar evidencias",
        "Definir impacto e escopo",
        "Reproduzir quando seguro",
        "Encaminhar para a fila adequada",
    ],
}


class InputModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class Ticket(InputModel):
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=5, max_length=5000)
    affected_users: int = Field(default=1, ge=1, le=100000)
    service_stopped: bool = False
    vip_or_critical_area: bool = False


class BatchIn(InputModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=100)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def priority(ticket: Ticket) -> str:
    if ticket.service_stopped and (
        ticket.affected_users >= 10 or ticket.vip_or_critical_area
    ):
        return "critical"
    if (
        ticket.service_stopped
        or ticket.affected_users >= 10
        or ticket.vip_or_critical_area
    ):
        return "high"
    if ticket.affected_users >= 3:
        return "medium"
    return "low"


def predict_category(text: str):
    probabilities = MODEL.predict_proba([normalize(text)])[0]
    pairs = sorted(
        zip(MODEL.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    category, confidence = pairs[0]
    alternatives = [
        {"category": str(label), "confidence": round(float(score), 3)}
        for label, score in pairs[:3]
    ]
    return str(category), round(float(confidence), 3), alternatives


def review_reasons(
    category: str,
    ticket_priority: str,
    confidence: float,
    alternatives: list[dict],
) -> list[str]:
    reasons = []
    if category == "security":
        reasons.append("security_category")
    if ticket_priority == "critical":
        reasons.append("critical_impact")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("low_model_confidence")
    if len(alternatives) > 1:
        margin = alternatives[0]["confidence"] - alternatives[1]["confidence"]
        if margin < LOW_MARGIN_THRESHOLD:
            reasons.append("ambiguous_top_predictions")
    return reasons


def triage_one(ticket: Ticket):
    text = f"{ticket.title} {ticket.description}"
    category, confidence, alternatives = predict_category(text)
    ticket_priority = priority(ticket)
    reasons = review_reasons(category, ticket_priority, confidence, alternatives)
    confidence_margin = (
        round(alternatives[0]["confidence"] - alternatives[1]["confidence"], 3)
        if len(alternatives) > 1
        else confidence
    )
    keywords = sorted(
        set(re.findall(r"[A-Za-z0-9_-]{4,}", normalize(text)))
    )[:20]

    return {
        "category": category,
        "priority": ticket_priority,
        "confidence": confidence,
        "confidence_margin": confidence_margin,
        "alternatives": alternatives,
        "keywords": keywords,
        "suggested_next_steps": PLAYBOOKS[category],
        "human_review_required": bool(reasons),
        "human_review_reasons": reasons,
        "model": MODEL_METRICS["selected_model"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    with open("frontend.html", "r", encoding="utf-8") as file:
        return file.read()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.1.0",
        "model": MODEL_METRICS["selected_model"],
    }


@app.get("/model/metrics")
def model_metrics():
    return read_metrics()


@app.get("/model/comparison")
def model_comparison():
    return read_comparison()


@app.get("/model/confusion")
def model_confusion():
    return read_confusion()


@app.post("/triage")
def triage(ticket: Ticket):
    return triage_one(ticket)


@app.post("/triage/batch")
def triage_batch(body: BatchIn):
    results = [triage_one(ticket) for ticket in body.tickets]
    review_count = sum(1 for result in results if result["human_review_required"])
    return {
        "total": len(results),
        "review_required": review_count,
        "by_category": dict(Counter(result["category"] for result in results)),
        "by_priority": dict(Counter(result["priority"] for result in results)),
        "results": results,
    }
