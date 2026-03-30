# AI-Generated Restaurant Review Detector

A DistilBERT classifier that detects AI-generated restaurant reviews, with an "adjusted rating" feature that recalculates a restaurant's star rating after stripping detected fakes.

## Architecture

- **Model:** Fine-tuned `distilbert-base-uncased` (PyTorch + HuggingFace Transformers)
- **Data:** ~5k real Yelp reviews + ~5k Claude-generated fakes
- **API:** FastAPI with a simple HTML frontend
- **Deploy:** Docker → GCP Cloud Run

## Project Structure

```
ai-review-detector/
├── data/
│   ├── raw/              # yelp_reviews.csv, fake_reviews.csv
│   └── processed/        # train.csv, val.csv, test.csv
├── models/               # saved model checkpoints
├── api/
│   ├── main.py           # FastAPI app
│   ├── model.py          # inference logic
│   └── templates/        # HTML frontend
├── notebooks/            # exploration & analysis
├── scripts/
│   ├── collect_yelp.py        # download & sample Yelp reviews
│   ├── generate_fake_reviews.py  # generate fakes via Claude API
│   └── prepare_dataset.py     # combine, label, split
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Data Collection

```bash
# Step 1: Download and sample real Yelp reviews (uses HuggingFace datasets)
python scripts/collect_yelp.py

# Step 2: Generate fake reviews with Claude API (~5000 reviews, takes ~30-60 min)
python scripts/generate_fake_reviews.py

# Step 3: Combine, label, and split into train/val/test
python scripts/prepare_dataset.py
```

**Output:**
- `data/raw/yelp_reviews.csv` — real reviews (label=0)
- `data/raw/fake_reviews.csv` — AI-generated reviews (label=1)
- `data/processed/train.csv`, `val.csv`, `test.csv`

## Model Training

```bash
python scripts/train.py
```

Trains for up to 5 epochs with early stopping. Best checkpoint saved to `models/best_model/`.

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000` for the HTML frontend, or use the endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Healthcheck |
| `/detect` | POST | Classify a single review |
| `/adjust-rating` | POST | Recalculate rating after stripping fakes |

### `/detect` example

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing tacos, will definitely come back!"}'
```

```json
{"label": "real", "confidence": 0.94, "is_ai": false}
```

### `/adjust-rating` example

```bash
curl -X POST http://localhost:8000/adjust-rating \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {"text": "Great food!", "star_rating": 5},
      {"text": "Worst experience ever.", "star_rating": 1}
    ]
  }'
```

```json
{
  "original_rating": 3.0,
  "adjusted_rating": 5.0,
  "num_flagged": 1,
  "total_reviews": 2
}
```

## Docker

```bash
docker build -t ai-review-detector .
docker run -p 8000:8000 ai-review-detector
```

## GCP Cloud Run Deploy

See Step 5 instructions (coming soon).
