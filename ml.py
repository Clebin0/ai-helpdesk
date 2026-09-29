from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"tickets.csv"
ARTIFACTS=ROOT/"artifacts"
MODEL_PATH=ARTIFACTS/"ticket_classifier.joblib"
METRICS_PATH=ARTIFACTS/"metrics.json"
MATRIX_PATH=ARTIFACTS/"confusion_matrix.csv"

def make_pipeline()->Pipeline:
    return Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),strip_accents="unicode",lowercase=True,min_df=1)),
        ("model",LogisticRegression(max_iter=1500,random_state=42,class_weight="balanced")),
    ])

def train(save:bool=True):
    df=pd.read_csv(DATA)
    x_train,x_test,y_train,y_test=train_test_split(
        df["text"],df["category"],test_size=.28,random_state=42,stratify=df["category"]
    )
    model=make_pipeline()
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    labels=sorted(df["category"].unique().tolist())
    report=classification_report(y_test,pred,labels=labels,output_dict=True,zero_division=0)
    metrics={
        "accuracy":round(float(accuracy_score(y_test,pred)),4),
        "macro_f1":round(float(report["macro avg"]["f1-score"]),4),
        "weighted_f1":round(float(report["weighted avg"]["f1-score"]),4),
        "samples":int(len(df)),
        "train_samples":int(len(x_train)),
        "test_samples":int(len(x_test)),
        "labels":labels,
        "note":"Synthetic portfolio dataset; metrics are not representative of production traffic.",
    }
    if save:
        ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump(model,MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        pd.DataFrame(confusion_matrix(y_test,pred,labels=labels),index=labels,columns=labels).to_csv(MATRIX_PATH)
    return model,metrics

def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    model,_=train(save=False)
    return model
