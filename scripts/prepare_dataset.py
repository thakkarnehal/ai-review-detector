"""
prepare_dataset.py
------------------
Combine real Yelp reviews + all fake sources into one labeled dataset,
then split 80/10/10 into train/val/test CSVs.

Inputs (all in data/raw/):
    yelp_reviews.csv   label=0, real
    fake_reviews.csv   label=1, Claude Haiku
    gpt_reviews.csv    label=1, GPT-4o  (optional — included if present)

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
RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
# ─────────────────────────────────────────────────────────────────────────────

COLS = ["review_id", "text", "star_rating", "label"]


def load_csv(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=COLS)
    df["source"] = source
    return df


def print_stats(name: str, df: pd.DataFrame):
    n = len(df)
    n_real = (df["label"] == 0).sum()
    n_fake = (df["label"] == 1).sum()
    print(f"  {name:10s}  {n:6,} rows  |  real={n_real:,}  fake={n_fake:,}  "
          f"({n_real/n*100:.1f}% / {n_fake/n*100:.1f}%)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load all sources ───────────────────────────────────────────────────────
    real_path  = os.path.join(RAW_DIR, "yelp_reviews.csv")
    haiku_path = os.path.join(RAW_DIR, "fake_reviews.csv")
    gpt_path   = os.path.join(RAW_DIR, "gpt_reviews.csv")

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Missing {real_path} — run scripts/collect_yelp.py first.")
    if not os.path.exists(haiku_path):
        raise FileNotFoundError(f"Missing {haiku_path} — run scripts/generate_fake_reviews.py first.")

    print("Loading data...")
    real_df  = load_csv(real_path,  source="yelp")
    haiku_df = load_csv(haiku_path, source="claude-haiku")

    fake_parts = [haiku_df]
    print(f"  Real (Yelp)         : {len(real_df):,}")
    print(f"  Fake (Claude Haiku) : {len(haiku_df):,}")

    if os.path.exists(gpt_path):
        gpt_df = load_csv(gpt_path, source="gpt-4o")
        fake_parts.append(gpt_df)
        print(f"  Fake (GPT-4o)       : {len(gpt_df):,}")
    else:
        print(f"  Fake (GPT-4o)       : not found — skipping ({gpt_path})")

    fake_df = pd.concat(fake_parts, ignore_index=True)

    # ── Balance classes ────────────────────────────────────────────────────────
    n_real, n_fake = len(real_df), len(fake_df)
    print(f"\nBefore balancing — real: {n_real:,}  fake: {n_fake:,}")

    if abs(n_real - n_fake) / max(n_real, n_fake) > 0.25:
        target = min(n_real, n_fake)
        print(f"Class imbalance > 25% — capping both to {target:,}.")
        real_df = real_df.sample(n=target, random_state=RANDOM_SEED)
        fake_df = fake_df.sample(n=target, random_state=RANDOM_SEED)

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    assert "text"  in combined.columns, "Missing 'text' column"
    assert "label" in combined.columns, "Missing 'label' column"
    assert combined["label"].isin([0, 1]).all(), "Labels must be 0 or 1"

    before = len(combined)
    combined = combined[combined["text"].notna() & (combined["text"].str.strip() != "")]
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} rows with empty text.")

    print(f"\nCombined dataset: {len(combined):,} reviews")
    print(f"  Real : {(combined['label']==0).sum():,}")
    print(f"  Fake : {(combined['label']==1).sum():,}")
    print("\nFake source breakdown:")
    print(combined[combined["label"]==1]["source"].value_counts().to_string())

    # ── Split 80/10/10 ─────────────────────────────────────────────────────────
    train_val, test = train_test_split(
        combined,
        test_size=TEST_RATIO,
        stratify=combined["label"],
        random_state=RANDOM_SEED,
    )
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["label"],
        random_state=RANDOM_SEED,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    train_path = os.path.join(OUT_DIR, "train.csv")
    val_path   = os.path.join(OUT_DIR, "val.csv")
    test_path  = os.path.join(OUT_DIR, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path,   index=False)
    test.to_csv(test_path,  index=False)

    print("\nSplit summary:")
    print_stats("train", train)
    print_stats("val",   val)
    print_stats("test",  test)

    # ── Stats file ─────────────────────────────────────────────────────────────
    stats_path = os.path.join(OUT_DIR, "dataset_stats.txt")
    combined["word_count"] = combined["text"].apply(lambda x: len(str(x).split()))
    with open(stats_path, "w") as f:
        f.write("Dataset Statistics\n" + "=" * 40 + "\n\n")
        f.write(f"Total reviews : {len(combined):,}\n")
        f.write(f"  Real (0)    : {(combined['label']==0).sum():,}\n")
        f.write(f"  Fake (1)    : {(combined['label']==1).sum():,}\n\n")
        f.write("Fake source breakdown:\n")
        f.write(combined[combined["label"]==1]["source"].value_counts().to_string())
        f.write(f"\n\nTrain : {len(train):,}\n")
        f.write(f"Val   : {len(val):,}\n")
        f.write(f"Test  : {len(test):,}\n\n")
        f.write("Word count stats:\n")
        f.write(combined["word_count"].describe().to_string())

    print(f"\nStats saved → {stats_path}")
    print(f"\nFiles written:")
    for p in [train_path, val_path, test_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
