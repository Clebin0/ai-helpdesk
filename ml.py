from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"tickets.csv"
ARTIFACTS=ROOT/"artifacts"
MODEL_PATH=ARTIFACTS/"ticket_classifier.joblib"
METRICS_PATH=ARTIFACTS/"metrics.json"
COMPARISON_PATH=ARTIFACTS/"model_comparison.json"
MATRIX_PATH=ARTIFACTS/"confusion_matrix.json"

def make_pipeline(estimator)->Pipeline:
    return Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),strip_accents="unicode",lowercase=True,sublinear_tf=True)),
        ("model",estimator),
    ])

def candidates()->dict[str,Pipeline]:
    return {
        "logistic_regression":make_pipeline(LogisticRegression(max_iter=1800,random_state=42,class_weight="balanced")),
        "complement_nb":make_pipeline(ComplementNB(alpha=.7)),
        "sgd_log_loss":make_pipeline(SGDClassifier(loss="log_loss",max_iter=1800,random_state=42,class_weight="balanced")),
    }

def compare_models(x,y)->list[dict]:
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]
    for name,model in candidates().items():
        scores=cross_validate(model,x,y,cv=cv,scoring={"accuracy":"accuracy","macro_f1":"f1_macro"},n_jobs=None)
        rows.append({
            "model":name,
            "cv_accuracy_mean":round(float(scores["test_accuracy"].mean()),4),
            "cv_accuracy_std":round(float(scores["test_accuracy"].std()),4),
            "cv_macro_f1_mean":round(float(scores["test_macro_f1"].mean()),4),
            "cv_macro_f1_std":round(float(scores["test_macro_f1"].std()),4),
        })
    return sorted(rows,key=lambda r:r["cv_macro_f1_mean"],reverse=True)

def train(save:bool=True):
    df=pd.read_csv(DATA)
    x_train,x_test,y_train,y_test=train_test_split(
        df["text"],df["category"],test_size=.30,random_state=42,stratify=df["category"]
    )
    comparison=compare_models(x_train,y_train)
    selected_name=comparison[0]["model"]
    model=candidates()[selected_name]
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    labels=sorted(df["category"].unique().tolist())
    report=classification_report(y_test,pred,labels=labels,output_dict=True,zero_division=0)
    matrix=confusion_matrix(y_test,pred,labels=labels)
    metrics={
        "selected_model":selected_name,
        "accuracy":round(float(accuracy_score(y_test,pred)),4),
        "macro_f1":round(float(report["macro avg"]["f1-score"]),4),
        "weighted_f1":round(float(report["weighted avg"]["f1-score"]),4),
        "samples":int(len(df)),
        "train_samples":int(len(x_train)),
        "test_samples":int(len(x_test)),
        "labels":labels,
        "per_class_f1":{label:round(float(report[label]["f1-score"]),4) for label in labels},
        "note":"Synthetic portfolio dataset. Holdout metrics do not estimate production performance.",
    }
    matrix_payload={"labels":labels,"matrix":matrix.astype(int).tolist()}
    if save:
        ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump(model,MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        COMPARISON_PATH.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
        MATRIX_PATH.write_text(json.dumps(matrix_payload,indent=2),encoding="utf-8")
    return model,metrics,comparison,matrix_payload

def ensure_artifacts():
    needed=[MODEL_PATH,METRICS_PATH,COMPARISON_PATH,MATRIX_PATH]
    if not all(p.exists() for p in needed):
        train(save=True)

def load_model():
    ensure_artifacts()
    return joblib.load(MODEL_PATH)

def read_metrics():
    ensure_artifacts()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def read_comparison():
    ensure_artifacts()
    return json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))

def read_confusion():
    ensure_artifacts()
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
