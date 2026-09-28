# AI Helpdesk

API de triagem de chamados com classificação local, explicável e opção de integração futura com LLM.

## O que faz
- Classifica categoria por sinais do texto
- Sugere prioridade a partir de impacto/urgência
- Identifica palavras-chave
- Gera próximos passos seguros
- Não executa comandos na máquina do usuário
- Endpoint FastAPI pronto para integração com front-end/help desk

## Executar
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Documentação: `/docs`.

## Categorias
Rede, hardware, software, acesso, impressão e outros.

## IA
A versão inicial usa regras transparentes para funcionar sem enviar dados a terceiros. Uma camada LLM pode ser adicionada depois com anonimização, controle de PII, aprovação humana e fallback determinístico.