"""
generate_gpt_reviews.py
-----------------------
Generate 2,500 fake restaurant reviews using GPT-4o with the same varied
prompts (star ratings, cuisines, cities, tones, lengths) as the Claude script.

Set OPENAI_API_KEY in .env before running.

Output: data/raw/gpt_reviews.csv
Columns: review_id, text, star_rating, cuisine, city, tone, length_target,
         generated_model, label (always 1 = AI-generated)

Usage:
    python scripts/generate_gpt_reviews.py
    python scripts/generate_gpt_reviews.py --count 500   # quick test
"""

import os
import csv
import time
import random
import hashlib
import argparse
from pathlib import Path

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_COUNT     = 2500
MODEL            = "gpt-4o"
MAX_RETRIES      = 3
RETRY_DELAY      = 5
RATE_LIMIT_DELAY = 0.5   # GPT-4o is slower than Haiku, give it a bit more room
CHECKPOINT_EVERY = 100
OUTPUT_PATH      = Path(__file__).parent.parent / "data" / "raw" / "gpt_reviews.csv"
# ─────────────────────────────────────────────────────────────────────────────

CUISINES = [
    "Italian", "Mexican", "Japanese", "Chinese", "Indian", "Thai",
    "American", "French", "Mediterranean", "Korean", "Vietnamese",
    "Greek", "Ethiopian", "Peruvian", "Middle Eastern", "BBQ",
    "Seafood", "Pizza", "Sushi", "Tacos", "Burgers", "Ramen",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Denver", "Seattle", "Nashville",
    "Portland", "Las Vegas", "Atlanta", "Miami", "Boston",
]

TONES = [
    "casual and conversational",
    "enthusiastic and exclamatory",
    "detailed and analytical",
    "brief and to the point",
    "disappointed and critical",
    "neutral and balanced",
    "humorous with sarcasm",
    "warm and personal",
    "formal and professional",
    "like a local regular",
]

LENGTH_TARGETS = {
    "short":  (20,  50),
    "medium": (60,  120),
    "long":   (140, 250),
}

SYSTEM_PROMPTS = [
    "You are a regular Yelp reviewer. Write reviews that sound authentic and personal, like they were typed on a phone.",
    "You write Google Maps reviews. Be direct, casual, and include specific details about your visit.",
    "You're a food blogger who occasionally posts short reviews. Use descriptive food language but keep it accessible.",
    "You're a frequent diner who reviews mostly to help others decide. Focus on practical details: wait time, value, service.",
    "You write TripAdvisor reviews. Mention context like who you went with, the occasion, and whether you'd return.",
]

STAR_WEIGHTS = {1: 0.12, 2: 0.08, 3: 0.13, 4: 0.30, 5: 0.37}


def make_review_id(text: str) -> str:
    return "gpt_" + hashlib.md5(text.encode()).hexdigest()[:10]


def build_user_prompt(star: int, cuisine: str, city: str, tone: str, length_key: str) -> str:
    min_w, max_w = LENGTH_TARGETS[length_key]
    sentiment = {
        1: "terrible — you had a very bad experience",
        2: "below average — a few okay things but mostly disappointing",
        3: "mixed — some good, some bad",
        4: "good — you enjoyed it with minor complaints",
        5: "excellent — you loved everything",
    }[star]
    return (
        f"Write a {star}-star restaurant review for a {cuisine} restaurant in {city}. "
        f"Your overall sentiment is {sentiment}. "
        f"Tone: {tone}. "
        f"Length: {min_w}–{max_w} words. "
        f"Do NOT include a title or star rating in the text — just write the review body. "
        f"Do NOT start with 'I' as the first word."
    )


def generate_review(client: openai.OpenAI, star: int, cuisine: str, city: str,
                    tone: str, length_key: str, system_prompt: str) -> str | None:
    user_prompt = build_user_prompt(star, cuisine, city, tone, length_key)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            tqdm.write(f"  Rate limited — waiting {wait}s...")
            time.sleep(wait)
        except openai.APIError as e:
            tqdm.write(f"  API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None


def load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_checkpoint(records: list[dict], path: Path):
    if not records:
        return
    fieldnames = ["review_id", "text", "star_rating", "cuisine", "city",
                  "tone", "length_target", "generated_model", "label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main(target_count: int):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Add it to .env or export it.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(api_key=api_key)

    records = load_checkpoint(OUTPUT_PATH)
    already_done = len(records)
    if already_done:
        print(f"Resuming from checkpoint: {already_done} reviews already generated.")

    remaining = target_count - already_done
    if remaining <= 0:
        print(f"Already have {already_done} reviews — nothing to do.")
        return

    print(f"Generating {remaining} reviews (target: {target_count})...")
    print(f"Model: {MODEL} | Output: {OUTPUT_PATH}\n")

    # Pre-build star pool with correct distribution
    stars_pool = []
    for star, weight in STAR_WEIGHTS.items():
        stars_pool.extend([star] * round(weight * target_count))
    random.shuffle(stars_pool)
    while len(stars_pool) < target_count:
        stars_pool.append(random.choices(
            list(STAR_WEIGHTS.keys()), weights=list(STAR_WEIGHTS.values())
        )[0])
    stars_pool = stars_pool[:target_count]

    errors = 0
    with tqdm(total=remaining, desc="Generating", unit="review") as pbar:
        for i in range(already_done, target_count):
            star       = stars_pool[i]
            cuisine    = random.choice(CUISINES)
            city       = random.choice(CITIES)
            tone       = random.choice(TONES)
            length_key = random.choices(
                list(LENGTH_TARGETS.keys()), weights=[0.25, 0.50, 0.25]
            )[0]
            system_prompt = random.choice(SYSTEM_PROMPTS)

            text = generate_review(client, star, cuisine, city, tone, length_key, system_prompt)

            if text is None:
                errors += 1
                tqdm.write(f"  Skipping #{i+1} after all retries. Total errors: {errors}")
                pbar.update(1)
                continue

            records.append({
                "review_id":      make_review_id(text),
                "text":           text,
                "star_rating":    star,
                "cuisine":        cuisine,
                "city":           city,
                "tone":           tone,
                "length_target":  length_key,
                "generated_model": MODEL,
                "label":          1,
            })

            pbar.update(1)
            pbar.set_postfix(errors=errors, saved=len(records))

            if len(records) % CHECKPOINT_EVERY == 0:
                save_checkpoint(records, OUTPUT_PATH)

            time.sleep(RATE_LIMIT_DELAY)

    save_checkpoint(records, OUTPUT_PATH)
    print(f"\nDone. {len(records)} reviews saved → {OUTPUT_PATH}")
    print(f"Errors/skipped: {errors}")

    import pandas as pd
    df = pd.DataFrame(records)
    print("\nStar rating distribution:")
    print(df["star_rating"].astype(int).value_counts().sort_index().to_string())
    print("\nCuisine distribution (top 10):")
    print(df["cuisine"].value_counts().head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=TARGET_COUNT,
                        help=f"Reviews to generate (default: {TARGET_COUNT})")
    args = parser.parse_args()
    main(args.count)
