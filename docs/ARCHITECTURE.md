# Arquitetura

A classificação do chamado e a prioridade ficam separadas. O modelo tenta identificar a categoria; impacto e urgência continuam sendo tratados por regras simples.

```mermaid
flowchart LR
    A[Chamado] --> B[FastAPI]
    B --> C[Normalização do texto]
    C --> D[TF-IDF]
    D --> E[Classificador]
    E --> F[Categoria e confiança]
    A --> G[Regras de impacto]
    G --> H[Prioridade]
    F --> I[Resultado]
    H --> I
```

## Treinamento

```mermaid
flowchart LR
    A[Dataset de chamados] --> B[Train / holdout]
    B --> C[Validação cruzada]
    C --> D[Logistic Regression]
    C --> E[Complement Naive Bayes]
    C --> F[SGD]
    D --> G[Comparação por Macro F1]
    E --> G
    F --> G
    G --> H[Modelo escolhido]
```

Chamados críticos, de segurança ou com confiança baixa continuam marcados para revisão manual. O dataset do repositório é de demonstração e não contém chamados reais.
