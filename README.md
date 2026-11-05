# Helpdesk Triage

Experimento de classificação de chamados de TI com machine learning tradicional. O objetivo é sugerir uma fila e destacar casos que não devem seguir sem análise de uma pessoa.

![Painel de triagem](docs/screenshots/dashboard.png)

## O que o classificador faz

O texto do título e da descrição passa por normalização e TF-IDF (termos isolados e pares de termos). Durante o treinamento, três classificadores são comparados:

- Logistic Regression
- Complement Naive Bayes
- SGD com perda logarítmica

A seleção usa Macro F1 médio em validação cruzada estratificada no conjunto de treino. Depois, o modelo escolhido é medido em um holdout separado. A API expõe as métricas, a comparação e a matriz de confusão para que o resultado não fique escondido atrás de uma única porcentagem.

**O classificador não é um chatbot e não usa um LLM.** Ele classifica texto e continua separado do novo módulo experimental LLM RAG Assistant, documentado em [docs/LLM_RAG.md](docs/LLM_RAG.md). O assistente recupera trechos de documentação e gera respostas com fontes; não altera chamados nem executa ações em infraestrutura.

## Dados usados no experimento

O arquivo `data/tickets.csv` contém 70 exemplos sintéticos: dez por categoria (`network`, `hardware`, `software`, `access`, `printing`, `security` e `other`). A divisão é estratificada, com 70% para treino e 30% para holdout. A validação cruzada acontece somente dentro da parte de treino.

Esse conjunto é pequeno e foi escrito para exercitar o pipeline. As métricas atuais servem para comparar os experimentos deste repositório; não devem ser interpretadas como desempenho esperado em uma central de atendimento real.

## Categoria e prioridade são coisas diferentes

O classificador sugere a categoria. A prioridade é calculada por regras operacionais a partir de indisponibilidade, quantidade de pessoas afetadas e indicação de área crítica.

A API exige revisão humana quando ocorre pelo menos uma destas condições:

- A categoria sugerida é segurança.
- O impacto calculado é crítico.
- A confiança prevista fica abaixo de 0,72.
- A diferença entre as duas categorias mais prováveis fica abaixo de 0,12.

Esses limites são parâmetros iniciais do laboratório, não uma política universal. A probabilidade retornada pelo classificador não é uma confiança calibrada nem garantia de acerto.

## Executar

Requisitos: Python 3.12 ou superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

Abra [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Estado da API e modelo carregado |
| GET | `/model/metrics` | Métricas do holdout |
| GET | `/model/comparison` | Resultado da validação cruzada |
| GET | `/model/confusion` | Matriz de confusão |
| POST | `/triage` | Classificar um chamado |
| POST | `/triage/batch` | Classificar até 100 chamados |

Exemplo de entrada:

```json
{
  "title": "Sem acesso à internet",
  "description": "O notebook conecta ao Wi-Fi, mas não resolve DNS.",
  "affected_users": 3,
  "service_stopped": false,
  "vip_or_critical_area": false
}
```

## Testes

```bash
python -m pytest -q
```

Os testes cobrem categorias conhecidas, cálculo de prioridade, encaminhamento obrigatório para revisão, triagem em lote e validação de entrada.

## O que falta para um uso real

O dataset precisa ser substituído ou complementado por chamados representativos, anonimizados e autorizados. Também faltam monitoramento de qualidade, calibração das probabilidades, análise de deriva, versionamento formal dos modelos e avaliação com analistas. Até isso existir, a saída deve ser tratada como sugestão.

## LLM & RAG Assistant (módulo experimental)

O repositório agora também contém um assistente separado do classificador tradicional, acessível pelo painel Assistente de IA na página inicial. Ele usa embeddings para recuperar trechos de documentação e a Responses API para redigir respostas com referências às fontes.

- Ingestão autenticada de documentos autorizados.
- Embeddings e busca semântica local em SQLite.
- Respostas fundamentadas no contexto recuperado, com citações [S1], [S2].
- Abstenção quando a recuperação não encontra material relevante.
- Interface web de chat com referências às fontes e token guardado apenas na sessão do navegador.
- Tokens separados para chat e administração; nenhuma chave é armazenada no código.

Consulte [docs/LLM_RAG.md](docs/LLM_RAG.md) para configuração, interface web, chamadas de API, segurança e limitações. Para executar o módulo com a API da OpenAI, você precisa fornecer sua própria chave e configurar os dois tokens de acesso. O uso da API pode gerar custos.
