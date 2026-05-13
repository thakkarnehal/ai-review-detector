"""
train.py
--------
Fine-tune distilbert-base-uncased for binary review classification.
  label 0 = real   label 1 = AI-generated

Usage:
    python scripts/train.py                     # full training + evaluation
    python scripts/train.py --eval-only         # skip training, eval best checkpoint
    python scripts/train.py --epochs 3 --batch-size 16
"""

import os
import csv
import json
import time
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "best_model")

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 256
THRESHOLD   = 0.5   # sigmoid threshold for predicting class 1


# ── Dataset ───────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["text"].fillna("").tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class DistilBertClassifier(nn.Module):
    """DistilBERT + single linear head + sigmoid for binary classification."""

    def __init__(self, pretrained: str, dropout: float = 0.3):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(pretrained)
        hidden_size     = self.bert.config.dim  # 768
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        dropped   = self.dropout(cls_token)
        logits    = self.classifier(dropped).squeeze(-1)  # (batch,)
        return logits  # raw logits — apply sigmoid for probabilities


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, threshold=THRESHOLD):
    preds = (np.array(probs) >= threshold).astype(int)
    acc   = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


# ── Train / eval passes ───────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    n_steps = len(loader)
    t0 = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

            if train and (step % 50 == 0 or step == n_steps):
                elapsed   = time.time() - t0
                remaining = (elapsed / step) * (n_steps - step)
                avg_loss  = total_loss / step
                print(
                    f"\r  step {step}/{n_steps} | loss={avg_loss:.4f} "
                    f"| ETA {remaining:.0f}s   ",
                    end="", flush=True
                )

    if train:
        print()  # newline after \r

    avg_loss = total_loss / n_steps
    metrics  = compute_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    return metrics, all_labels, all_probs


# ── Plots & artifacts ─────────────────────────────────────────────────────────

def save_confusion_matrix(labels, probs, out_path):
    preds = (np.array(probs) >= THRESHOLD).astype(int)
    cm    = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Real", "AI"], yticklabels=["Real", "AI"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out_path}")


def save_training_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_f1"],       label="Val F1")
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Val metrics per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {out_path}")


def save_misclassifications(test_df, labels, probs, out_path, n_each=10):
    """Save n_each false positives + n_each false negatives to CSV."""
    preds = (np.array(probs) >= THRESHOLD).astype(int)
    df    = test_df.copy().reset_index(drop=True)
    df["true_label"]  = labels
    df["pred_label"]  = preds
    df["confidence"]  = probs

    fp = df[(df["true_label"] == 0) & (df["pred_label"] == 1)].head(n_each)
    fn = df[(df["true_label"] == 1) & (df["pred_label"] == 0)].head(n_each)
    misclf = pd.concat([fp, fn], ignore_index=True)
    misclf["error_type"] = ["false_positive"] * len(fp) + ["false_negative"] * len(fn)

    misclf[["review_id", "text", "true_label", "pred_label", "confidence", "error_type"]].to_csv(
        out_path, index=False
    )
    print(f"Misclassifications saved → {out_path}  (FP={len(fp)}, FN={len(fn)})")


def save_confidence_distribution(labels, probs, out_path):
    """Plot confidence distributions for correct vs incorrect predictions."""
    preds   = (np.array(probs) >= THRESHOLD).astype(int)
    labels  = np.array(labels)
    correct = np.array(probs)[labels == preds]
    wrong   = np.array(probs)[labels != preds]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct, bins=40, alpha=0.6, label=f"Correct (n={len(correct)})", color="steelblue")
    ax.hist(wrong,   bins=40, alpha=0.6, label=f"Incorrect (n={len(wrong)})",  color="tomato")
    ax.axvline(THRESHOLD, color="black", linestyle="--", label=f"Threshold={THRESHOLD}")
    ax.set_xlabel("Model confidence (P(AI))")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution: correct vs incorrect predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confidence distribution saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(epochs: int, batch_size: int, lr: float, patience: int, eval_only: bool):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"Train label dist: {train_df['label'].value_counts().to_dict()}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # pos_weight to handle class imbalance (n_real / n_fake)
    n_real = (train_df["label"] == 0).sum()
    n_fake = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_real / n_fake], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if not eval_only:
        train_ds = ReviewDataset(train_df, tokenizer, MAX_LEN)
        val_ds   = ReviewDataset(val_df,   tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True)

        model = DistilBertClassifier(MODEL_NAME).to(device)

        total_steps  = len(train_loader) * epochs
        warmup_steps = total_steps // 10
        optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_f1      = 0.0
        epochs_no_improve = 0
        history = {k: [] for k in ["train_loss", "val_loss", "val_accuracy",
                                    "val_precision", "val_recall", "val_f1", "val_auc_roc"]}

        print(f"Training for up to {epochs} epochs (early stopping patience={patience})...\n")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            t_epoch = time.time()

            train_metrics, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, train=True
            )
            val_metrics, _, _   = run_epoch(
                model, val_loader, criterion, None, None, device, train=False
            )

            epoch_time = time.time() - t_epoch
            print(
                f"  train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_prec={val_metrics['precision']:.4f} | "
                f"val_rec={val_metrics['recall']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_auc={val_metrics['auc_roc']:.4f} | "
                f"time={epoch_time:.1f}s"
            )

            for k in history:
                src = train_metrics if k.startswith("train") else val_metrics
                key = k.replace("train_", "").replace("val_", "")
                history[k].append(src[key])

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                epochs_no_improve = 0
                # Save model weights + tokenizer
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model.pt"))
                tokenizer.save_pretrained(CHECKPOINT_DIR)
                print(f"  ✓ Best val F1={best_val_f1:.4f} — checkpoint saved")
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        # Save training history
        history_path = os.path.join(MODELS_DIR, "training_metrics.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved → {history_path}")

        save_training_curves(history, os.path.join(MODELS_DIR, "training_curves.png"))

    # ── Evaluation on test set ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("TEST SET EVALUATION")
    print("="*55)

    test_ds     = ReviewDataset(test_df, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    best_model = DistilBertClassifier(MODEL_NAME).to(device)
    ckpt_path  = os.path.join(CHECKPOINT_DIR, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run training first.")
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}\n")

    test_metrics, test_labels, test_probs = run_epoch(
        best_model, test_loader, criterion, None, None, device, train=False
    )

    preds = (np.array(test_probs) >= THRESHOLD).astype(int)

    print("Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=["Real", "AI-generated"]))

    # Confidence distribution print
    correct_conf = np.array(test_probs)[np.array(test_labels) == preds]
    wrong_conf   = np.array(test_probs)[np.array(test_labels) != preds]
    print(f"Confidence (correct predictions) — mean={correct_conf.mean():.3f} std={correct_conf.std():.3f}")
    print(f"Confidence (wrong predictions)   — mean={wrong_conf.mean():.3f} std={wrong_conf.std():.3f}")

    # Artifacts
    save_confusion_matrix(test_labels, test_probs,
                          os.path.join(MODELS_DIR, "confusion_matrix.png"))
    save_confidence_distribution(test_labels, test_probs,
                                 os.path.join(MODELS_DIR, "confidence_distribution.png"))
    save_misclassifications(test_df, test_labels, test_probs,
                            os.path.join(MODELS_DIR, "misclassifications.csv"))

    # Save test metrics
    test_metrics_path = os.path.join(MODELS_DIR, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in test_metrics.items()}, f, indent=2)
    print(f"Test metrics saved → {test_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--patience",   type=int,   default=2)
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, load best checkpoint and run test evaluation only")
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.patience, args.eval_only)
