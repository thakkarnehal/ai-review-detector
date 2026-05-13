"""
stress_test.py
--------------
Generate 200 harder-to-detect reviews and evaluate the trained model on them.

  50 — Claude claude-sonnet-4-6, adversarial prompt (typos, casual, informal)
  50 — GPT-4o, adversarial prompt
  50 — Claude Haiku (same model as training data, different adversarial prompt)
  50 — Real reviews from Yelp/yelp_review_full TEST split (unseen during training)

Outputs:
  data/stress_test/stress_test_reviews.csv   all 200 reviews + true labels
  data/stress_test/stress_test_results.csv   + model predictions + confidence
  data/stress_test/stress_test_report.txt    per-source breakdown
  data/stress_test/stress_test_confusion.png confusion matrix

Usage:
    python scripts/stress_test.py
    python scripts/stress_test.py --eval-only   # skip generation, re-run eval
"""

import os
import time
import random
import hashlib
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
from transformers import AutoTokenizer, DistilBertModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)
import anthropic
import openai
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "best_model")
OUT_DIR        = os.path.join(BASE_DIR, "data", "stress_test")
MAX_LEN        = 256
THRESHOLD      = 0.5
BATCH_SIZE     = 16
N_PER_SOURCE   = 50
RATE_DELAY     = 0.5   # seconds between API calls
RANDOM_SEED    = 99
# ─────────────────────────────────────────────────────────────────────────────

CUISINES = ["Italian", "Mexican", "Japanese", "American", "Thai", "Indian",
            "Korean", "Mediterranean", "BBQ", "Seafood"]
CITIES   = ["New York", "Chicago", "Los Angeles", "Austin", "Seattle",
            "Miami", "Denver", "Nashville", "Portland", "Boston"]
STARS    = [1, 2, 3, 4, 5]

# Adversarial system prompt — same for Claude and GPT
ADVERSARIAL_SYSTEM = (
    "You write casual, authentic-sounding Yelp reviews. "
    "Your reviews sound like they were typed fast on a phone by a real person. "
    "Use informal language, occasional typos, abbreviations, run-on sentences, "
    "filler words like 'like' and 'honestly' and 'tbh', and realistic imperfections. "
    "Never sound polished or formal. Vary your style significantly between reviews."
)

ADVERSARIAL_USER_TEMPLATE = (
    "Write a {star}-star Yelp review for a {cuisine} restaurant in {city}. "
    "Make it sound like a real casual person wrote it — typos are fine, keep it natural. "
    "{extra} "
    "Just write the review body, no title."
)

EXTRA_INSTRUCTIONS = [
    "Keep it short, under 60 words.",
    "Write a longer rant, 100-150 words.",
    "Start with a personal anecdote.",
    "Mention something specific like a dish name or a waiter.",
    "Use some ALL CAPS for emphasis.",
    "Include a comparison to another restaurant.",
    "Mention you came with friends or family.",
    "Ask a rhetorical question in the review.",
]


def make_id(text: str, prefix: str) -> str:
    return f"{prefix}_" + hashlib.md5(text.encode()).hexdigest()[:10]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_claude(n: int, model: str, source_label: str) -> list[dict]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    records = []
    print(f"  Generating {n} reviews with {model}...")

    for i in range(n):
        star    = random.choice(STARS)
        cuisine = random.choice(CUISINES)
        city    = random.choice(CITIES)
        extra   = random.choice(EXTRA_INSTRUCTIONS)
        prompt  = ADVERSARIAL_USER_TEMPLATE.format(
            star=star, cuisine=cuisine, city=city, extra=extra
        )
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=300,
                    system=ADVERSARIAL_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                break
            except anthropic.RateLimitError:
                time.sleep(5 * (attempt + 1))
                text = None
            except anthropic.APIError as e:
                print(f"    API error: {e}")
                text = None
                break

        if text:
            records.append({
                "review_id":   make_id(text, source_label),
                "text":        text,
                "star_rating": star,
                "source":      source_label,
                "label":       1,
            })
        time.sleep(RATE_DELAY)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} done")

    return records


def generate_gpt4o(n: int) -> list[dict]:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    records = []
    print(f"  Generating {n} reviews with gpt-4o...")

    for i in range(n):
        star    = random.choice(STARS)
        cuisine = random.choice(CUISINES)
        city    = random.choice(CITIES)
        extra   = random.choice(EXTRA_INSTRUCTIONS)
        prompt  = ADVERSARIAL_USER_TEMPLATE.format(
            star=star, cuisine=cuisine, city=city, extra=extra
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=300,
                    messages=[
                        {"role": "system", "content": ADVERSARIAL_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content.strip()
                break
            except openai.RateLimitError:
                time.sleep(5 * (attempt + 1))
                text = None
            except openai.APIError as e:
                print(f"    API error: {e}")
                text = None
                break

        if text:
            records.append({
                "review_id":   make_id(text, "gpt4o"),
                "text":        text,
                "star_rating": star,
                "source":      "gpt-4o",
                "label":       1,
            })
        time.sleep(RATE_DELAY)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} done")

    return records


def sample_real_yelp(n: int) -> list[dict]:
    """Sample from the Yelp test split — completely unseen during training."""
    print(f"  Sampling {n} real reviews from Yelp test split...")
    dataset = load_dataset("Yelp/yelp_review_full", split="test")
    df = dataset.to_pandas()  # cols: label, text
    df = df.rename(columns={"label": "label_raw"})

    df["wc"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[(df["wc"] >= 20) & (df["wc"] <= 300)].copy()

    sampled = df.sample(n=n, random_state=RANDOM_SEED)
    records = []
    for _, row in sampled.iterrows():
        records.append({
            "review_id":   make_id(row["text"], "yelp_test"),
            "text":        row["text"],
            "star_rating": int(row["label_raw"]) + 1,
            "source":      "yelp-test-split",
            "label":       0,
        })
    return records


# ── Model (copied from train.py) ──────────────────────────────────────────────

class DistilBertClassifier(nn.Module):
    def __init__(self, pretrained: str, dropout: float = 0.3):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(pretrained)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.dim, 1)

    def forward(self, input_ids, attention_mask):
        out     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls     = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls)).squeeze(-1)


class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = [str(t) for t in texts]
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
        }


def run_inference(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_and_save_report(df: pd.DataFrame, report_path: str):
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    labels = df["label"].values
    probs  = df["confidence"].values
    preds  = df["predicted"].values

    log("=" * 60)
    log("STRESS TEST RESULTS — OVERALL")
    log("=" * 60)
    acc  = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(labels, probs)
    log(f"  Accuracy  : {acc:.4f}")
    log(f"  Precision : {prec:.4f}")
    log(f"  Recall    : {rec:.4f}")
    log(f"  F1        : {f1:.4f}")
    log(f"  AUC-ROC   : {auc:.4f}")
    log()
    log(classification_report(labels, preds, target_names=["Real", "AI-generated"]))

    log("=" * 60)
    log("PER-SOURCE BREAKDOWN")
    log("=" * 60)
    for source, grp in df.groupby("source"):
        src_labels = grp["label"].values
        src_preds  = grp["predicted"].values
        src_probs  = grp["confidence"].values
        src_acc    = accuracy_score(src_labels, src_preds)
        # flagged as AI
        flagged    = src_preds.sum()
        true_label = src_labels[0]
        kind       = "AI (should be flagged)" if true_label == 1 else "Real (should NOT be flagged)"
        log(f"\n  {source}  [{kind}]")
        log(f"    n={len(grp)}  flagged_as_AI={flagged}  accuracy={src_acc:.3f}")
        log(f"    avg confidence (P(AI)): {src_probs.mean():.3f}  ±{src_probs.std():.3f}")
        if true_label == 1:
            missed = (src_preds == 0).sum()
            log(f"    missed (false negatives): {missed} / {len(grp)}")
        else:
            wrong = (src_preds == 1).sum()
            log(f"    wrongly flagged (false positives): {wrong} / {len(grp)}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {report_path}")


def save_confusion(labels, preds, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "AI"], yticklabels=["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Stress Test")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(eval_only: bool):
    random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    reviews_path = os.path.join(OUT_DIR, "stress_test_reviews.csv")
    results_path = os.path.join(OUT_DIR, "stress_test_results.csv")

    # ── Generation ─────────────────────────────────────────────────────────────
    if not eval_only:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set in .env")
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set in .env")

        print("Generating adversarial reviews...\n")

        all_records = []

        print("[1/4] Claude claude-sonnet-4-6 (adversarial)")
        all_records += generate_claude(N_PER_SOURCE, "claude-sonnet-4-6", "claude-sonnet")

        print("\n[2/4] GPT-4o (adversarial)")
        all_records += generate_gpt4o(N_PER_SOURCE)

        print("\n[3/4] Claude Haiku (adversarial — same model as training, different prompts)")
        all_records += generate_claude(N_PER_SOURCE, "claude-haiku-4-5", "claude-haiku-adversarial")

        print("\n[4/4] Real Yelp reviews (test split — unseen)")
        all_records += sample_real_yelp(N_PER_SOURCE)

        df = pd.DataFrame(all_records)
        df.to_csv(reviews_path, index=False)
        print(f"\nAll {len(df)} reviews saved → {reviews_path}")
    else:
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"No reviews file at {reviews_path}. Run without --eval-only first.")
        df = pd.read_csv(reviews_path)
        print(f"Loaded {len(df)} reviews from {reviews_path}")

    # ── Inference ──────────────────────────────────────────────────────────────
    print("\nRunning model inference...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model     = DistilBertClassifier("distilbert-base-uncased").to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train the model first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    dataset = ReviewDataset(df["text"].tolist(), tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probs = run_inference(model, loader, device)
    preds = (probs >= THRESHOLD).astype(int)

    df["confidence"] = probs
    df["predicted"]  = preds
    df.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}\n")

    # ── Report ─────────────────────────────────────────────────────────────────
    print_and_save_report(df, os.path.join(OUT_DIR, "stress_test_report.txt"))
    save_confusion(df["label"].values, preds, os.path.join(OUT_DIR, "stress_test_confusion.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, load existing reviews CSV and re-run eval")
    args = parser.parse_args()
    main(args.eval_only)
