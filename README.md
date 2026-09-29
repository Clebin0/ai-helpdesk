# AI Helpdesk ML

Projeto de portfólio para triagem de chamados de TI com **Machine Learning local e explicável**.

## O que mudou na V3

A classificação deixou de depender somente de regras. O projeto agora treina um pipeline real de NLP:

- `TfidfVectorizer` com unigramas e bigramas
- `LogisticRegression` com balanceamento de classes
- divisão treino/teste estratificada
- accuracy, macro F1 e weighted F1
- matriz de confusão exportada
- probabilidades usadas como sinal de confiança
- revisão humana para baixa confiança, incidentes críticos e segurança

A prioridade continua baseada em regras de impacto/urgência porque categoria e severidade são problemas diferentes.

## Dataset

`data/tickets.csv` contém exemplos **sintéticos** criados exclusivamente para demonstração. As métricas desse conjunto não devem ser interpretadas como desempenho em produção.

## Treinamento

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

Isso gera localmente:

- `artifacts/ticket_classifier.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.csv`

## Aplicação

```bash
uvicorn app:app --reload
```

Abra `http://127.0.0.1:8000`.

## API

- `POST /triage`
- `POST /triage/batch`
- `GET /model/metrics`
- `GET /health`

## Testes

```bash
python -m pytest -q
```

## Limitações

O dataset é pequeno e sintético. O objetivo é demonstrar pipeline, avaliação, serving e decisões de produto; não afirmar precisão de produção. Um deployment real exigiria dados anonimizados, análise de drift, avaliação por fila/idioma, observabilidade e governança.
