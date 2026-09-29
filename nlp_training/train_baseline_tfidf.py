"""
Trains a TF-IDF + Logistic Regression intent classifier.

This is the "fast baseline" model: no GPU, no internet download, trains in
seconds even on a laptop CPU. Useful in a report as a comparison point
against both the rule-based regex parser and a fine-tuned transformer —
i.e. "does the extra complexity of a transformer actually buy us anything
over a classic ML baseline?" is a genuinely interesting question to answer
with real numbers.

Usage:
    python train_baseline_tfidf.py

Reads:  data/train.csv, data/val.csv
Writes: models/tfidf_baseline.joblib
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # unigrams + bigrams capture short phrases
                # like "go to" / "move to" that a single-word model would miss
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                C=5.0,
                class_weight="balanced",
            ),
        ),
    ])

    print(f"Training on {len(train_df)} examples...")
    pipeline.fit(train_df["text"], train_df["label"])

    val_preds = pipeline.predict(val_df["text"])
    val_acc = accuracy_score(val_df["label"], val_preds)
    val_f1 = f1_score(val_df["label"], val_preds, average="macro")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro-F1: {val_f1:.4f}")

    model_path = MODEL_DIR / "tfidf_baseline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model -> {model_path}")

    metrics = {"val_accuracy": val_acc, "val_macro_f1": val_f1, "train_size": len(train_df)}
    with open(MODEL_DIR / "tfidf_baseline_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()