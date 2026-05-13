# AI-Generated Restaurant Review Detector

A DistilBERT classifier that detects AI-generated restaurant reviews, with an "adjusted rating" feature that recalculates a restaurant's star rating after stripping detected fakes.

## Architecture

- **Model:** Fine-tuned `distilbert-base-uncased` (PyTorch + HuggingFace Transformers)
- **Data:** ~5k real Yelp reviews + ~5k AI-generated fakes (Claude + GPT-4o)
- **API:** FastAPI served on port 8080
- **Deploy:** Docker → AWS ECR → ECS Fargate
- **Extension:** Chrome extension for Yelp and TripAdvisor

## Project Structure

```
ai-review-detector/
├── data/
│   ├── raw/              # yelp_reviews.csv, fake_reviews.csv, gpt_reviews.csv
│   ├── processed/        # train.csv, val.csv, test.csv
│   └── stress_test/      # stress test results
├── models/               # checkpoints, metrics, confusion matrix
├── api/
│   ├── main.py           # FastAPI app
│   ├── model.py          # inference logic
│   └── templates/        # HTML frontend
├── extension/            # Chrome extension (Manifest V3)
├── scripts/
│   ├── collect_yelp.py        # download & sample Yelp reviews
│   ├── generate_fake_reviews.py  # generate fakes via Claude API
│   ├── generate_gpt_reviews.py   # generate fakes via GPT-4o
│   ├── prepare_dataset.py        # combine, label, split
│   └── stress_test.py            # evaluate on held-out adversarial set
├── requirements.txt
└── Dockerfile
```

## Model Performance

Evaluated on 200 held-out reviews (50 real Yelp, 50 Claude Haiku adversarial, 50 Claude Sonnet, 50 GPT-4o):

| Metric | Score |
|---|---|
| Accuracy | 98.5% |
| Precision | 100% |
| Recall | 98.0% |
| F1 | 98.99% |
| AUC-ROC | 100% |

Zero false positives on real Yelp reviews. Misses ~1 in 50 AI reviews across all model sources.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set ANTHROPIC_API_KEY and OPENAI_API_KEY in .env
```

## Data Collection

```bash
python scripts/collect_yelp.py          # real Yelp reviews via HuggingFace datasets
python scripts/generate_fake_reviews.py # Claude-generated fakes
python scripts/generate_gpt_reviews.py  # GPT-4o-generated fakes
python scripts/prepare_dataset.py       # combine, label, train/val/test split
```

## Training

```bash
python scripts/train.py
```

Trains for up to 5 epochs with early stopping. Best checkpoint saved to `models/best_model/`.

## Running the API Locally

```bash
uvicorn api.main:app --reload --port 8080
```

Open `http://localhost:8080` for the HTML frontend.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/detect` | POST | Classify a single review |
| `/adjust-rating` | POST | Recalculate rating after stripping fakes |

### `/detect`

```bash
curl -X POST http://localhost:8080/detect \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Amazing tacos, will definitely come back!"}'
```

```json
{"label": "Real", "confidence": 0.02, "flagged": false}
```

### `/adjust-rating`

```bash
curl -X POST http://localhost:8080/adjust-rating \
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
  "flagged_count": 1,
  "total_reviews": 2
}
```

## Docker

```bash
# Build for local (Apple Silicon)
docker build -t ai-review-detector .

# Build for AWS (required for ECS Fargate)
docker buildx build --platform linux/amd64 -t ai-review-detector .
```

## AWS Deployment (ECS Fargate)

The API is deployed to ECS Fargate in `us-east-1`. To redeploy after changes:

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 724680459083.dkr.ecr.us-east-1.amazonaws.com

# Build and push (must use linux/amd64 for Fargate)
docker buildx build --platform linux/amd64 -t review-detector .
docker tag review-detector:latest \
  724680459083.dkr.ecr.us-east-1.amazonaws.com/review-detector:latest
docker push 724680459083.dkr.ecr.us-east-1.amazonaws.com/review-detector:latest

# Force new deployment
aws ecs update-service --cluster review-detector \
  --service review-detector --force-new-deployment
```

The Fargate task is assigned a public IP on each deployment. After redeployment, update the `API_URL` in `extension/content.js` and `extension/popup.js` to the new IP.

## Chrome Extension

Located in `extension/`. Works on `yelp.com/biz/*` and `tripadvisor.com/Restaurant*`.

**To load in Chrome:**
1. Go to `chrome://extensions`
2. Enable Developer mode
3. Click "Load unpacked" → select the `extension/` folder

The extension auto-scans reviews on page load, adds a badge to each review (✓ authentic / 🚩 AI-generated), and shows a floating widget with the adjusted star rating.

## Limitations

- **HTTP only:** The API runs over HTTP. Chrome logs a mixed-content warning when the extension calls it from HTTPS Yelp/TripAdvisor pages. The extension works via a background service worker that bypasses the block, but the proper fix is HTTPS (requires a domain + ALB).
- **Dynamic IP:** The Fargate task gets a new public IP on every deployment. The extension's API URL must be manually updated after each redeploy.
- **Training data skew:** The model was trained primarily on Claude-generated fakes. It generalizes well to GPT-4o (98% accuracy in stress tests) but may be less robust against future models or heavily human-edited AI text.
- **Yelp DOM selectors:** The Yelp scraping logic targets CSS class patterns that can break if Yelp updates their frontend.
- **No auth on the API:** The `/detect` endpoint is publicly accessible with no rate limiting. Fine for personal use, not for production.
- **Review length floor:** Reviews under 80 characters are skipped by the extension (too short to classify reliably).

## Next Steps

- **HTTPS:** Add an ALB with an ACM certificate to serve the API over HTTPS and eliminate the mixed-content warning. Also gives a stable DNS name so the extension URL never needs updating.
- **Rate limiting:** Add per-IP rate limiting to the API (e.g. slowapi) to prevent abuse.
- **Expand training data:** Add more AI model sources (Gemini, Llama) and human-edited AI text to make the classifier more robust.
- **TripAdvisor support:** The extension has TripAdvisor selectors but they're untested — validate and fix the DOM targeting.
- **Chrome Web Store:** Package and publish the extension so others can install it without loading unpacked.
- **Retrain on failure cases:** Review `models/misclassifications.csv` and add those examples back into training.
