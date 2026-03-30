"""
collect_yelp.py
---------------
Download Yelp reviews from HuggingFace (Yelp/yelp_review_full) and sample
~6,000 real restaurant reviews balanced across star ratings 1-5.

The HF dataset is a cleaned subset of the Yelp Open Dataset with 650k reviews.
It does NOT include business-type metadata, so we cannot filter to restaurants
only — but the reviews are general Yelp reviews, the majority of which are for
restaurants/food businesses, making it a strong proxy.

Output: data/raw/yelp_reviews.csv
Columns: review_id, text, star_rating, word_count, label (always 0 = real)
"""

import os
import hashlib
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_PER_STAR = 1200   # 1200 × 5 stars = 6,000 total
MIN_WORDS = 20
MAX_WORDS = 300
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "yelp_reviews.csv")
# ─────────────────────────────────────────────────────────────────────────────


def word_count(text: str) -> int:
    return len(text.split())


def make_review_id(text: str) -> str:
    return "yelp_" + hashlib.md5(text.encode()).hexdigest()[:10]


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading Yelp/yelp_review_full from HuggingFace (this downloads ~500 MB the first time)...")
    # The dataset has a 'train' split (650k) and 'test' split (50k).
    # Labels are 0–4 corresponding to star ratings 1–5.
    dataset = load_dataset("Yelp/yelp_review_full", split="train")

    print(f"Loaded {len(dataset):,} reviews. Sampling {TARGET_PER_STAR} per star rating...")

    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()
    df.columns = ["text", "label_raw"]  # HF cols: text, label (0-indexed stars)
    df["star_rating"] = df["label_raw"] + 1  # shift to 1-5
    df["word_count"] = df["text"].apply(word_count)

    # Filter by length
    df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)].copy()
    print(f"After length filter ({MIN_WORDS}–{MAX_WORDS} words): {len(df):,} reviews")

    # Sample balanced across stars
    sampled_parts = []
    for star in range(1, 6):
        subset = df[df["star_rating"] == star]
        n = min(TARGET_PER_STAR, len(subset))
        sampled = subset.sample(n=n, random_state=RANDOM_SEED)
        sampled_parts.append(sampled)
        print(f"  ★{star}: sampled {n:,} / {len(subset):,} available")

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    # Build final dataframe
    out = pd.DataFrame({
        "review_id": result["text"].apply(make_review_id),
        "text": result["text"],
        "star_rating": result["star_rating"],
        "word_count": result["word_count"],
        "label": 0,  # 0 = real
    })

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out):,} reviews → {OUTPUT_PATH}")
    print("\nStar rating distribution:")
    print(out["star_rating"].value_counts().sort_index().to_string())
    print(f"\nWord count stats:\n{out['word_count'].describe().to_string()}")


if __name__ == "__main__":
    main()
