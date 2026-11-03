# AI Helpdesk ML

Triagem explicável de chamados de suporte com NLP, comparação de modelos e revisão humana orientada por risco.

## Visão geral

O projeto classifica o texto de um chamado em categorias de suporte usando um pipeline TF-IDF. Os candidatos são comparados por Macro F1 médio em validação cruzada estratificada. O melhor modelo é avaliado em um holdout separado e servido por uma API FastAPI.

## Demonstração

![AI Helpdesk dashboard](docs/screenshots/dashboard.png)

As métricas exibidas foram calculadas sobre o dataset sintético versionado no repositório e não representam desempenho de produção.

## Funcionalidades

- Classificação em network, hardware, software, access, printing, security e other.
- Comparação entre Logistic Regression, Complement Naive Bayes e SGD com log loss.
- Métricas de Accuracy, Macro F1, Weighted F1 e F1 por categoria.
- Matriz de confusão e comparação dos modelos.
- Prioridade operacional calculada separadamente da categoria prevista.
- Top 3 previsões, confiança e margem entre as duas principais classes.
- Motivos explícitos para exigir revisão humana: categoria de segurança, impacto crítico, baixa confiança ou predições ambíguas.
- Triagem em lote com contagem dos casos que exigem revisão.
- Playbooks de próximos passos por categoria.

## Stack

- Python 3.12+
- FastAPI e Pydantic
- scikit-learn, pandas e joblib
- pytest e HTTPX

## Executar localmente

Windows PowerShell:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
~~~

Abra http://127.0.0.1:8000.

## Testes

~~~bash
python -m pytest -q
python -m compileall -q .
~~~

A suíte inclui previsões conhecidas, escalonamento de segurança, revisão por ambiguidade, impacto crítico, triagem em lote e validação de entradas.

## API principal

| Método | Rota | Objetivo |
|---|---|---|
| GET | /health | Estado e modelo selecionado |
| GET | /model/metrics | Métricas do holdout |
| GET | /model/comparison | Comparação de modelos |
| GET | /model/confusion | Matriz de confusão |
| POST | /triage | Triagem individual |
| POST | /triage/batch | Triagem de até 100 chamados |

Exemplo de requisição:

~~~json
{
  "title": "Sem acesso à internet",
  "description": "O notebook conecta ao Wi-Fi, mas não resolve DNS.",
  "affected_users": 3,
  "service_stopped": false,
  "vip_or_critical_area": false
}
~~~

## Avaliação responsável

O dataset é sintético e pequeno. As métricas servem para demonstrar o pipeline e comparar experimentos; não estimam desempenho em ambiente real.

Confiança do classificador não é garantia de correção. Chamados de segurança, críticos, ambíguos ou com baixa confiança são sinalizados para revisão humana. A API não envia respostas automaticamente a clientes nem executa ações em infraestrutura.

## Dados e privacidade

Não use tickets, nomes, e-mails, identificadores, logs ou conteúdo de empresas reais. Para experiências com dados reais, anonimize a origem e valide base legal, controle de acesso, retenção e segurança antes de treinar qualquer modelo.
