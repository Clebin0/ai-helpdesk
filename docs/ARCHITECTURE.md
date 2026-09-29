# Architecture

```mermaid
flowchart LR
    A[Synthetic ticket dataset] --> B[Train / test split]
    B --> C[TF-IDF]
    C --> D[Logistic Regression]
    D --> E[Metrics + confusion matrix]
    D --> F[Serialized model]
    F --> G[FastAPI inference]
    G --> H[Category + confidence]
    H --> I[Priority rules]
    I --> J[Human review]
```

The classifier predicts the ticket category. Priority is intentionally handled by a separate deterministic policy using impact and outage information; classification confidence is not treated as incident severity.
