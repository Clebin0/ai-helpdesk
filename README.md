# AI Helpdesk ML

Projeto de portfólio para triagem explicável de chamados de TI usando Machine Learning local.

## Pipeline

O projeto compara três pipelines de NLP no conjunto de treino:

- TF-IDF + Logistic Regression
- TF-IDF + Complement Naive Bayes
- TF-IDF + SGD Classifier com log loss

A seleção usa Macro F1 médio em validação cruzada estratificada. Depois, o modelo escolhido é avaliado em um holdout separado.

## O que é medido

- Accuracy
- Macro F1
- Weighted F1
- F1 por categoria
- Matriz de confusão
- Probabilidades para o top 3 de classes

O dataset em `data/tickets.csv` é pequeno e sintético. As métricas demonstram o pipeline e não estimam desempenho em produção.

## Produto

A aplicação web mostra:
- comparação entre modelos;
- métricas do modelo selecionado;
- matriz de confusão;
- triagem de um chamado;
- prioridade calculada separadamente por impacto;
- revisão humana obrigatória para baixa confiança, segurança ou incidente crítico.

## Executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

Abra `http://127.0.0.1:8000`.

## API

- `POST /triage`
- `POST /triage/batch`
- `GET /model/metrics`
- `GET /model/comparison`
- `GET /model/confusion`
- `GET /health`

## Limitações

Não há dados corporativos reais neste repositório. Um uso real exigiria dataset anonimizado, avaliação por fila e idioma, calibração, drift monitoring, auditoria e controles de acesso.
