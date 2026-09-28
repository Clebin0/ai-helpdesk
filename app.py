from __future__ import annotations
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app=FastAPI(title="AI Helpdesk",version="2.0.0")

RULES={
"network":["internet","rede","wifi","wi-fi","dns","ip","cabo","conexao","vpn"],
"hardware":["ssd","memoria","monitor","teclado","mouse","computador","notebook","hd","bateria"],
"software":["erro","aplicativo","programa","windows","travando","sistema","dll","atualizacao"],
"access":["senha","login","acesso","bloqueado","permissao","mfa","conta"],
"printing":["impressora","imprimir","scanner","toner","fila"],
"security":["phishing","malware","virus","suspeito","fraude","credencial"]
}

PLAYBOOKS={
"network":["Confirmar escopo e usuarios afetados","Validar link, IP e gateway","Testar DNS e destino","Registrar evidencias antes de alterar configuracoes"],
"hardware":["Registrar modelo e sintomas","Verificar recursos e eventos do sistema","Executar testes nao destrutivos","Planejar troca somente apos diagnostico"],
"software":["Registrar mensagem de erro","Reproduzir o problema","Validar versao, logs e dependencias","Testar correcao em escopo controlado"],
"access":["Confirmar identidade pelo processo oficial","Validar conta, bloqueio e permissao","Evitar compartilhar credenciais","Escalar privilegios ao responsavel"],
"printing":["Validar energia e conectividade","Verificar fila e impressora padrao","Testar pagina de diagnostico","Registrar erro do driver ou spooler"],
"security":["Isolar o risco sem apagar evidencias","Orientar o usuario a nao interagir mais","Registrar origem, horario e indicadores","Escalar ao processo de seguranca"],
"other":["Coletar evidencias","Definir impacto e escopo","Reproduzir quando seguro","Encaminhar para a fila adequada"]
}

class Ticket(BaseModel):
    title:str=Field(min_length=3,max_length=160)
    description:str=Field(min_length=5,max_length=5000)
    affected_users:int=Field(default=1,ge=1,le=100000)
    service_stopped:bool=False
    vip_or_critical_area:bool=False

class BatchIn(BaseModel):
    tickets:list[Ticket]=Field(min_length=1,max_length=100)

def normalize(text:str)->str:
    value=unicodedata.normalize("NFKD",text.lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))

def classify(text:str):
    t=normalize(text)
    scores={k:sum(1 for w in words if w in t) for k,words in RULES.items()}
    category=max(scores,key=scores.get)
    confidence=min(0.99,0.45+0.12*scores[category]) if scores[category] else 0.25
    return (category if scores[category] else "other"),scores,round(confidence,2)

def priority(t:Ticket)->str:
    if t.service_stopped and (t.affected_users>=10 or t.vip_or_critical_area): return "critical"
    if t.service_stopped or t.affected_users>=10 or t.vip_or_critical_area: return "high"
    if t.affected_users>=3: return "medium"
    return "low"

def triage_one(ticket:Ticket):
    text=f"{ticket.title} {ticket.description}"
    category,scores,confidence=classify(text)
    keywords=sorted(set(re.findall(r"[A-Za-z0-9_-]{4,}",normalize(text))))[:20]
    return {
      "category":category,
      "priority":priority(ticket),
      "confidence":confidence,
      "confidence_signal":scores,
      "keywords":keywords,
      "suggested_next_steps":PLAYBOOKS[category],
      "human_review_required":True,
      "generated_at":datetime.now(timezone.utc).isoformat()
    }

@app.get("/",response_class=HTMLResponse)
def home():
    with open("frontend.html","r",encoding="utf-8") as f: return f.read()

@app.get("/health")
def health(): return {"status":"ok","version":"2.0.0"}

@app.post("/triage")
def triage(ticket:Ticket): return triage_one(ticket)

@app.post("/triage/batch")
def triage_batch(body:BatchIn):
    results=[triage_one(t) for t in body.tickets]
    return {"total":len(results),"by_category":dict(Counter(r["category"] for r in results)),"by_priority":dict(Counter(r["priority"] for r in results)),"results":results}
