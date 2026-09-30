# Architecture

## Goal

AI Helpdesk is a local ML triage pipeline for classifying IT tickets while keeping priority rules and human review explicit.

## Request flow

```mermaid
flowchart LR
    A[Ticket] --> B[FastAPI]
    B --> C[Text normalization]
    C --> D[TF-IDF pipeline]
    D --> E[Selected classifier]
    E --> F[Category + confidence]
    A --> G[Operational impact rules]
    G --> H[Priority]
    F --> I[Human review policy]
    H --> I
    I --> J[Triage response]
```

## Training flow

```mermaid
flowchart LR
    A[Synthetic ticket dataset] --> B[Train / holdout split]
    B --> C[Cross-validation]
    C --> D[Logistic Regression]
    C --> E[Complement Naive Bayes]
    C --> F[SGD log-loss]
    D --> G[Macro F1 comparison]
    E --> G
    F --> G
    G --> H[Selected model]
    H --> I[Serialized artifact + metrics]
```

## Design choices

Classification confidence never replaces operational impact. Critical tickets, security tickets and low-confidence predictions require human review. The repository contains synthetic data only.
