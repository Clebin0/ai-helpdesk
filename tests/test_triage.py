from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
def test_network_triage():
    r=client.post("/triage",json={"title":"Sem internet","description":"Usuário sem conexão de rede","affected_users":1})
    assert r.status_code==200
    assert r.json()["category"]=="network"
    assert r.json()["human_review_required"] is True
