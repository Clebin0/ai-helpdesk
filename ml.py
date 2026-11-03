from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "tickets.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "ticket_classifier.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
COMPARISON_PATH = ARTIFACTS / "model_comparison.json"
MATRIX_PATH = ARTIFACTS / "confusion_matrix.json"


def make_pipeline(estimator: Any) -> Pipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vectorizer), ("model", estimator)])


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": make_pipeline(
            LogisticRegression(
                max_iter=1800,
                random_state=42,
                class_weight="balanced",
            )
        ),
        "complement_nb": make_pipeline(ComplementNB(alpha=0.7)),
        "sgd_log_loss": make_pipeline(
            SGDClassifier(
                loss="log_loss",
                max_iter=1800,
                random_state=42,
                class_weight="balanced",
            )
        ),
    }


def compare_models(texts, labels) -> list[dict[str, Any]]:
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }
    results = []

    for name, pipeline in candidates().items():
        scores = cross_validate(
            pipeline,
            texts,
            labels,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=None,
        )
        results.append(
            {
                "model": name,
                "cv_accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
                "cv_accuracy_std": round(float(scores["test_accuracy"].std()), 4),
                "cv_macro_f1_mean": round(float(scores["test_macro_f1"].mean()), 4),
                "cv_macro_f1_std": round(float(scores["test_macro_f1"].std()), 4),
            }
        )

    return sorted(
        results,
        key=lambda row: row["cv_macro_f1_mean"],
        reverse=True,
    )


def train(save: bool = True):
    dataset = pd.read_csv(DATA)
    train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
        dataset["text"],
        dataset["category"],
        test_size=0.30,
        random_state=42,
        stratify=dataset["category"],
    )

    comparison = compare_models(train_texts, train_labels)
    selected_name = comparison[0]["model"]
    model = candidates()[selected_name]
    model.fit(train_texts, train_labels)

    predictions = model.predict(holdout_texts)
    class_names = sorted(dataset["category"].unique().tolist())
    report = classification_report(
        holdout_labels,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        holdout_labels,
        predictions,
        labels=class_names,
    )

    metrics = {
        "selected_model": selected_name,
        "accuracy": round(float(accuracy_score(holdout_labels, predictions)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "samples": int(len(dataset)),
        "train_samples": int(len(train_texts)),
        "test_samples": int(len(holdout_texts)),
        "labels": class_names,
        "per_class_f1": {
            label: round(float(report[label]["f1-score"]), 4)
            for label in class_names
        },
        "note": (
            "Synthetic portfolio dataset. Holdout metrics do not estimate "
            "production performance."
        ),
    }
    confusion_data = {
        "labels": class_names,
        "matrix": confusion.astype(int).tolist(),
    }

    if save:
        ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        COMPARISON_PATH.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        MATRIX_PATH.write_text(
            json.dumps(confusion_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return model, metrics, comparison, confusion_data


def ensure_artifacts() -> None:
    required_files = (
        MODEL_PATH,
        METRICS_PATH,
        COMPARISON_PATH,
        MATRIX_PATH,
    )
    if not all(path.exists() for path in required_files):
        train(save=True)


def load_model():
    ensure_artifacts()
    return joblib.load(MODEL_PATH)


def read_metrics() -> dict[str, Any]:
    ensure_artifacts()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def read_comparison() -> list[dict[str, Any]]:
    ensure_artifacts()
    return json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))


def read_confusion() -> dict[str, Any]:
    ensure_artifacts()
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
