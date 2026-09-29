"""
Fine-tunes a pretrained DistilBERT model (distilbert-base-uncased) for
intent classification on the generated dataset.

This is the "real" fine-tuning path for the report — a pretrained language
model adapted to this specific 12-class intent task via supervised
fine-tuning of a classification head (+ the full model, since dataset sizes
here are small enough to fine-tune all layers).

Requirements (not needed for the rest of the project — install separately):
    pip install transformers datasets torch scikit-learn accelerate

This needs internet access on first run (to download the pretrained model
from the HuggingFace Hub, ~260MB) and benefits from a GPU but will run on
CPU too — with ~1000 short training examples and a few epochs, CPU training
typically takes on the order of 10-20 minutes on a laptop.

Usage:
    python finetune_transformer.py [--epochs 8] [--batch-size 16] [--lr 2e-5]

Reads:  data/train.csv, data/val.csv, data/label_list.json
Writes: models/distilbert_intent/  (fine-tuned model + tokenizer)
        models/distilbert_train_history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_NAME = "distilbert-base-uncased"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing dependency for fine-tuning: "
            f"{e}\nInstall with: pip install transformers datasets torch accelerate"
        )

    label_list = json.loads((DATA_DIR / "label_list.json").read_text())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    train_df["label_id"] = train_df["label"].map(label2id)
    val_df["label_id"] = val_df["label"].map(label2id)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=32)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision_weighted": precision,
            "recall_weighted": recall,
        }

    output_dir = MODEL_DIR / "distilbert_intent"
    training_args_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],  # disable wandb/etc. — keep this self-contained
    )
    try:
        # transformers renamed evaluation_strategy= to eval_strategy= at
        # some point; try the current name first, fall back to the old one.
        training_args = TrainingArguments(eval_strategy="epoch", **training_args_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    try:
        # transformers >= 4.46 renamed Trainer's tokenizer= argument to
        # processing_class= (deprecating the old name entirely in some
        # versions, so we can't just pass both / rely on a warning).
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    print(f"Fine-tuning {args.model_name} on {len(train_df)} examples "
          f"for {args.epochs} epochs...")
    train_result = trainer.train()

    eval_metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    history = {
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "final_eval_metrics": eval_metrics,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "model_name": args.model_name,
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
    with open(MODEL_DIR / "distilbert_train_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nFine-tuned model saved -> {output_dir}")
    print("Run evaluate_models.py next to get test-set metrics and comparison tables.")


if __name__ == "__main__":
    main()