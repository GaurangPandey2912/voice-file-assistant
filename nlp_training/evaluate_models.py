"""
Evaluates every available intent-detection approach on the held-out test
set and produces a comparison table plus per-model confusion matrices —
the numbers to drop straight into a report's evaluation section.

Evaluates (whichever are available):
  1. Rule-based regex/spaCy parser  (voice_assistant.intent.parser)
  2. TF-IDF + Logistic Regression   (models/tfidf_baseline.joblib)
  3. Fine-tuned DistilBERT          (models/distilbert_intent/), if trained

Usage:
    python evaluate_models.py

Reads:  data/test.csv, data/label_list.json,
        models/tfidf_baseline.joblib (optional),
        models/distilbert_intent/ (optional)
Writes: results/comparison_table.csv
        results/confusion_matrix_<model>.png (per model)
        results/classification_report_<model>.txt (per model)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — just save PNGs, don't try to open a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"

# Make the project's own package importable so we can evaluate the real
# rule-based parser, not a reimplementation of it.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_labels() -> list[str]:
    return json.loads((DATA_DIR / "label_list.json").read_text())


def _predict_regex_parser(texts: list[str]) -> list[str]:
    from voice_assistant.intent.parser import IntentParser

    parser = IntentParser()
    preds = []
    for text in texts:
        result = parser.parse(text)
        preds.append(result.intent.name)
    return preds


def _predict_tfidf(texts: list[str]) -> list[str] | None:
    model_path = MODEL_DIR / "tfidf_baseline.joblib"
    if not model_path.exists():
        print(f"[skip] TF-IDF model not found at {model_path} "
              f"(run train_baseline_tfidf.py first)")
        return None
    import joblib
    pipeline = joblib.load(model_path)
    return list(pipeline.predict(texts))


def _predict_transformer(texts: list[str]) -> list[str] | None:
    model_dir = MODEL_DIR / "distilbert_intent"
    if not model_dir.exists():
        print(f"[skip] Fine-tuned transformer not found at {model_dir} "
              f"(run finetune_transformer.py first)")
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("[skip] transformers/torch not installed; can't evaluate the "
              "fine-tuned model. Install with: pip install transformers torch")
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
            logits = model(**inputs).logits
            pred_id = int(logits.argmax(dim=-1))
            preds.append(model.config.id2label[pred_id])
    return preds


def _report(name: str, y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro precision:   {prec_macro:.4f}")
    print(f"Macro recall:      {rec_macro:.4f}")
    print(f"Macro F1:          {f1_macro:.4f}")
    print(f"Weighted F1:       {f1_weighted:.4f}")

    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_path = RESULTS_DIR / f"classification_report_{name.lower().replace(' ', '_')}.txt"
    report_path.write_text(report_text)
    print(f"Full per-class report -> {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot -> {cm_path}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


def main() -> None:
    labels = _load_labels()
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    texts = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    print(f"Evaluating on {len(texts)} held-out test examples across {len(labels)} classes.\n")

    rows = []

    print("Running rule-based regex/spaCy parser...")
    regex_preds = _predict_regex_parser(texts)
    rows.append(_report("Regex Parser (baseline)", y_true, regex_preds, labels))

    print("\nRunning TF-IDF + Logistic Regression...")
    tfidf_preds = _predict_tfidf(texts)
    if tfidf_preds is not None:
        rows.append(_report("TF-IDF LogisticRegression", y_true, tfidf_preds, labels))

    print("\nRunning fine-tuned DistilBERT (if available)...")
    transformer_preds = _predict_transformer(texts)
    if transformer_preds is not None:
        rows.append(_report("Fine-tuned DistilBERT", y_true, transformer_preds, labels))

    comparison_df = pd.DataFrame(rows)
    comparison_path = RESULTS_DIR / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (also saved to results/comparison_table.csv)")
    print("=" * 70)
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()