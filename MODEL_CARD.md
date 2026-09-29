# Model Card — AI Helpdesk ML

## Model

- Task: multiclass text classification for IT helpdesk categories
- Pipeline: TF-IDF (1-2 grams) + Logistic Regression
- Language: Portuguese
- Classes: network, hardware, software, access, printing, security, other
- Serving: FastAPI

## Data

The repository uses a small **synthetic dataset** created only for portfolio demonstration. It does not contain company tickets, user data, credentials or production logs.

## Intended use

Demonstrate an end-to-end ML workflow: dataset preparation, training, evaluation, model persistence, inference API, confidence output, tests and CI.

## Not intended for

- autonomous incident resolution
- access decisions
- production security classification
- employee evaluation
- replacement of a support analyst

## Evaluation

Training exports accuracy, macro F1, weighted F1 and a confusion matrix. These values measure performance only on the synthetic holdout set and must not be interpreted as expected production accuracy.

## Human oversight

Human review is required for low-confidence predictions, security-related tickets and critical incidents.

## Production gaps

A real deployment would require anonymized historical data, data-quality checks, drift monitoring, per-class error analysis, calibration, retraining policy, audit logs and access controls.
