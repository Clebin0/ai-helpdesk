from fastapi.testclient import TestClient

import app as app_module
from app import Ticket, app, predict_category, review_reasons
from ml import read_comparison, read_confusion, read_metrics

client = TestClient(app)


def test_health_and_model_metadata():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model"] in {
        "logistic_regression",
        "complement_nb",
        "sgd_log_loss",
    }


def test_network_prediction():
    category, confidence, alternatives = predict_category(
        "computador sem internet e dns nao resolve"
    )
    assert category == "network"
    assert 0 <= confidence <= 1
    assert alternatives[0]["category"] == category


def test_security_prediction_requires_review():
    response = client.post(
        "/triage",
        json={
            "title": "Email suspeito",
            "description": "Usuario clicou em link de phishing e informou a senha",
            "affected_users": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "security"
    assert data["human_review_required"] is True
    assert "security_category" in data["human_review_reasons"]


def test_low_margin_requires_human_review():
    reasons = review_reasons(
        "network",
        "low",
        0.84,
        [
            {"category": "network", "confidence": 0.84},
            {"category": "software", "confidence": 0.79},
        ],
    )
    assert reasons == ["ambiguous_top_predictions"]


def test_critical_impact_requires_human_review():
    ticket = Ticket(
        title="Servico indisponivel",
        description="Sistema principal fora do ar para muitos usuarios",
        affected_users=100,
        service_stopped=True,
    )
    reasons = review_reasons(
        "network",
        "critical",
        0.99,
        [
            {"category": "network", "confidence": 0.99},
            {"category": "software", "confidence": 0.01},
        ],
    )
    assert "critical_impact" in reasons


def test_batch_reports_review_count():
    response = client.post(
        "/triage/batch",
        json={
            "tickets": [
                {
                    "title": "Erro de rede",
                    "description": "Computador sem conexao e DNS nao resolve",
                    "affected_users": 1,
                },
                {
                    "title": "Email suspeito",
                    "description": "Link de phishing e senha informada pelo usuario",
                    "affected_users": 1,
                },
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["review_required"] >= 1


def test_model_artifacts_are_explainable():
    metrics = read_metrics()
    comparison = read_comparison()
    matrix = read_confusion()
    assert metrics["samples"] >= 50
    assert len(comparison) == 3
    assert len(matrix["labels"]) == len(matrix["matrix"])


def test_ticket_rejects_blank_and_unknown_fields():
    blank = client.post(
        "/triage",
        json={"title": "   ", "description": "   "},
    )
    unknown = client.post(
        "/triage",
        json={
            "title": "Falha de rede",
            "description": "Sem internet no computador",
            "internal_customer_data": "not-allowed",
        },
    )
    assert blank.status_code == 422
    assert unknown.status_code == 422
