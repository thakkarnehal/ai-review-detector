"""
prepare_dataset.py
------------------
Combine real Yelp reviews + AI-generated fakes into one labeled dataset,
then split 80/10/10 into train/val/test CSVs.

Inputs:
    data/raw/yelp_reviews.csv  (label=0, real)
    data/raw/fake_reviews.csv  (label=1, AI-generated)

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/dataset_stats.txt
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
# ─────────────────────────────────────────────────────────────────────────────


def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["review_id", "text", "star_rating", "label"])
    df["source"] = "yelp"
    return df


def load_fake(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["review_id", "text", "star_rating", "label"])
    df["source"] = "claude"
    return df


def print_stats(name: str, df: pd.DataFrame):
    n = len(df)
    n_real = (df["label"] == 0).sum()
    n_fake = (df["label"] == 1).sum()
    print(f"  {name:10s}  {n:6,} rows  |  real={n_real:,}  fake={n_fake:,}  "
          f"({n_real/n*100:.1f}% / {n_fake/n*100:.1f}%)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    real_path = os.path.join(RAW_DIR, "yelp_reviews.csv")
    fake_path = os.path.join(RAW_DIR, "fake_reviews.csv")

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Missing {real_path} — run scripts/collect_yelp.py first.")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing {fake_path} — run scripts/generate_fake_reviews.py first.")

    print("Loading data...")
    real_df = load_real(real_path)
    fake_df = load_fake(fake_path)

    print(f"  Real reviews : {len(real_df):,}")
    print(f"  Fake reviews : {len(fake_df):,}")

    # Balance classes if significantly skewed (cap the larger to 2× the smaller)
    n_real, n_fake = len(real_df), len(fake_df)
    if abs(n_real - n_fake) / max(n_real, n_fake) > 0.25:
        target = min(n_real, n_fake)
        print(f"\nClass imbalance detected. Capping both classes to {target:,} for balance.")
        real_df = real_df.sample(n=target, random_state=RANDOM_SEED)
        fake_df = fake_df.sample(n=target, random_state=RANDOM_SEED)

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Validate required columns
    assert "text" in combined.columns, "Missing 'text' column"
    assert "label" in combined.columns, "Missing 'label' column"
    assert combined["label"].isin([0, 1]).all(), "Labels must be 0 or 1"

    # Drop rows with null/empty text
    before = len(combined)
    combined = combined[combined["text"].notna() & (combined["text"].str.strip() != "")]
    dropped = before - len(combined)
    if dropped:
        print(f"Dropped {dropped} rows with empty text.")

    print(f"\nCombined dataset: {len(combined):,} reviews")

    # ── Split ──────────────────────────────────────────────────────────────────
    # First split off test set, then split remainder into train/val
    train_val, test = train_test_split(
        combined,
        test_size=TEST_RATIO,
        stratify=combined["label"],
        random_state=RANDOM_SEED,
    )
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        stratify=train_val["label"],
        random_state=RANDOM_SEED,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    train_path = os.path.join(OUT_DIR, "train.csv")
    val_path   = os.path.join(OUT_DIR, "val.csv")
    test_path  = os.path.join(OUT_DIR, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print("\nSplit summary:")
    print_stats("train", train)
    print_stats("val", val)
    print_stats("test", test)

    # ── Stats file ────────────────────────────────────────────────────────────
    stats_path = os.path.join(OUT_DIR, "dataset_stats.txt")
    with open(stats_path, "w") as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total reviews : {len(combined):,}\n")
        f.write(f"  Real (0)    : {(combined['label']==0).sum():,}\n")
        f.write(f"  Fake (1)    : {(combined['label']==1).sum():,}\n\n")
        f.write(f"Train : {len(train):,}\n")
        f.write(f"Val   : {len(val):,}\n")
        f.write(f"Test  : {len(test):,}\n\n")
        f.write("Star rating distribution (combined):\n")
        f.write(combined["star_rating"].value_counts().sort_index().to_string())
        f.write("\n\nWord count stats:\n")
        combined["word_count"] = combined["text"].apply(lambda x: len(str(x).split()))
        f.write(combined["word_count"].describe().to_string())

    print(f"\nStats saved → {stats_path}")
    print(f"\nFiles written:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
