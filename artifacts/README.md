# Generated model artifacts

Run:

```bash
python train.py
```

The training process generates:

- `ticket_classifier.joblib`
- `metrics.json`
- `confusion_matrix.csv`

The serialized model is ignored by Git so binary training outputs do not need to be versioned. Reproducibility comes from the versioned dataset, training code and fixed random seed.
