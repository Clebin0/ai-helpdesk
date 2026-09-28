from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
def test_network_triage():
    r=client.post("/triage",json={"title":"Sem internet","description":"Usuario sem conexao de rede","affected_users":1})
    assert r.status_code==200 and r.json()["category"]=="network" and r.json()["human_review_required"] is True
