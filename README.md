# AI Helpdesk

Aplicacao de portfolio para triagem explicavel de chamados de TI.

## Recursos

- Interface web responsiva
- Classificacao por categoria
- Priorizacao por impacto e indisponibilidade
- Sinal de confianca explicavel
- Palavras-chave do chamado
- Playbooks por categoria
- Processamento em lote pela API
- Revisao humana obrigatoria
- Nenhum comando e executado no endpoint

## Categorias

Rede, hardware, software, acesso, impressao, seguranca e outros.

## Executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Abra http://127.0.0.1:8000. A documentacao interativa fica em /docs.

## Testes

```bash
python -m pytest -q
```

## Evolucao com LLM

A camada atual e deterministica para manter explicabilidade e privacidade. Uma integracao futura com LLM deve anonimizar PII, exigir aprovacao humana, registrar auditoria e manter fallback local.

## Seguranca

Use somente dados ficticios ou devidamente autorizados em demonstracoes publicas.
