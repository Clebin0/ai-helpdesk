from fastapi.testclient import TestClient
from app import app
client=TestClient(app)

def test_health():
    assert client.get("/health").status_code==200

def test_network_triage():
    r=client.post("/triage",json={"title":"Sem internet","description":"Usuario sem conexao de rede e DNS","affected_users":1})
    assert r.status_code==200
    data=r.json()
    assert data["category"]=="network"
    assert data["human_review_required"] is True

def test_critical_priority():
    r=client.post("/triage",json={"title":"Sistema parado","description":"Sistema principal indisponivel","affected_users":30,"service_stopped":True})
    assert r.json()["priority"]=="critical"
