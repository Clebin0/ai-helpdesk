from fastapi.testclient import TestClient
from app import app, predict_category
from ml import read_comparison, read_confusion, read_metrics
client=TestClient(app)

def test_health_and_model_metadata():
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["model"] in {"logistic_regression","complement_nb","sgd_log_loss"}

def test_network_prediction():
    category,confidence,_=predict_category("computador sem internet e dns nao resolve")
    assert category=="network"
    assert 0 <= confidence <= 1

def test_security_prediction_requires_review():
    r=client.post("/triage",json={"title":"Email suspeito","description":"Usuario clicou em link de phishing e informou a senha","affected_users":1})
    assert r.status_code==200
    data=r.json()
    assert data["category"]=="security"
    assert data["human_review_required"] is True

def test_model_artifacts_are_explainable():
    metrics=read_metrics(); comparison=read_comparison(); matrix=read_confusion()
    assert metrics["samples"] >= 50
    assert len(comparison)==3
    assert len(matrix["labels"])==len(matrix["matrix"])
