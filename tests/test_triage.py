from fastapi.testclient import TestClient
from app import app, predict_category
client=TestClient(app)

def test_health():
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["model"]=="tfidf-logistic-regression-v1"

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

def test_critical_priority():
    r=client.post("/triage",json={"title":"Sistema parado","description":"Aplicacao principal indisponivel para toda operacao","affected_users":30,"service_stopped":True})
    assert r.json()["priority"]=="critical"
