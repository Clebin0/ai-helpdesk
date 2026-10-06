<div align="center">

# AI Helpdesk ML

**NLP · Machine Learning · FastAPI · Explainability**

<img src="https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-000000?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/scikit--learn-000000?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />

</div>

---

Projeto de portfólio para triagem explicável de chamados de TI usando Machine Learning local.

## Pipeline

O laboratório compara três pipelines de NLP:

- TF-IDF + Logistic Regression;
- TF-IDF + Complement Naive Bayes;
- TF-IDF + SGD Classifier com log loss.

A seleção usa Macro F1 médio em validação cruzada estratificada. O modelo escolhido é avaliado em um holdout separado antes de ser servido pela API.

## Métricas

- Accuracy
- Macro F1
- Weighted F1
- F1 por categoria
- Matriz de confusão
- Top 3 probabilidades por classe

## Aplicação

A interface apresenta comparação de modelos, métricas, matriz de confusão e triagem de chamados. A prioridade é calculada separadamente por impacto e casos de baixa confiança, segurança ou incidentes críticos exigem revisão humana.

## API

- <code>POST /triage</code>
- <code>POST /triage/batch</code>
- <code>GET /model/metrics</code>
- <code>GET /model/comparison</code>
- <code>GET /model/confusion</code>
- <code>GET /health</code>

## Executar

~~~bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
~~~

## Limitações

O dataset é pequeno e sintético. As métricas demonstram o pipeline e não representam desempenho de produção. Um uso real exigiria dados anonimizados, calibração, monitoramento de drift, auditoria e controles de acesso.

---

<div align="center">

Projeto de laboratório para demonstrar engenharia aplicada de NLP.

</div>
