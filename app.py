from fastapi import FastAPI
from pydantic import BaseModel, Field
import re
app=FastAPI(title="AI Helpdesk",version="1.0.0")
RULES={
"network":["internet","rede","wifi","wi-fi","dns","ip","cabo","conexão","conexao"],
"hardware":["ssd","memória","memoria","monitor","teclado","mouse","computador","notebook"],
"software":["erro","aplicativo","programa","windows","travando","sistema"],
"access":["senha","login","acesso","bloqueado","permissão","permissao"],
"printing":["impressora","imprimir","scanner","toner"]
}
class Ticket(BaseModel):
    title:str=Field(min_length=3,max_length=160)
    description:str=Field(min_length=5,max_length=5000)
    affected_users:int=Field(default=1,ge=1,le=100000)
    service_stopped:bool=False

def classify(text):
    t=text.lower(); scores={k:sum(1 for w in ws if w in t) for k,ws in RULES.items()}
    category=max(scores,key=scores.get)
    return category if scores[category] else "other",scores

def priority(t:Ticket):
    if t.service_stopped and t.affected_users>=10:return "critical"
    if t.service_stopped or t.affected_users>=10:return "high"
    if t.affected_users>=3:return "medium"
    return "low"

@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/triage")
def triage(ticket:Ticket):
    text=f"{ticket.title} {ticket.description}"
    category,scores=classify(text)
    keywords=sorted(set(re.findall(r"[A-Za-zÀ-ÿ0-9_-]{4,}",text.lower())))[:20]
    guidance={
      "network":["Confirmar escopo do incidente","Validar link/interface e endereço IP","Testar gateway, DNS e destino"],
      "hardware":["Coletar sintomas e modelo do equipamento","Verificar recursos e eventos do sistema","Testar componente sem substituir dados do usuário"],
      "software":["Registrar mensagem de erro","Reproduzir o problema","Validar versão, logs e dependências"],
      "access":["Confirmar identidade pelo processo oficial","Validar conta e escopo da permissão","Escalar alterações privilegiadas para responsável"],
      "printing":["Validar energia/rede da impressora","Verificar fila e dispositivo padrão","Testar página de diagnóstico"],
      "other":["Coletar evidências","Definir impacto e escopo","Encaminhar à fila adequada"]
    }[category]
    return {"category":category,"priority":priority(ticket),"confidence_signal":scores,"keywords":keywords,"suggested_next_steps":guidance,"human_review_required":True}
