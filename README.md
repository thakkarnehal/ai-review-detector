# AI-Generated Restaurant Review Detector

> **Status: Decommissioned.** The API is offline. Kept as a reference and retrospective.

A DistilBERT classifier that detects AI-generated restaurant reviews, with a Chrome extension that badges reviews on Yelp and TripAdvisor and recalculates star ratings after stripping flagged ones.

## What it does

- Fine-tunes `distilbert-base-uncased` with a raw PyTorch training loop (no HuggingFace Trainer) to classify reviews as real or AI-generated
- Generates a labeled dataset: ~5k real Yelp reviews + ~5k AI fakes from Claude Haiku and GPT-4o, with varied prompts across star ratings, cuisines, tones, and lengths
- Serves a FastAPI inference API deployed on AWS ECS Fargate via Docker
- Chrome extension that auto-scans Yelp and TripAdvisor pages, adds a badge per review, and shows an adjusted star rating in a floating widget

## Why I stopped

The core problem is the task itself. AI detection is fundamentally adversarial and the approach doesn't scale to it.

**The model learned prompt artifacts, not AI writing.** All fake reviews were generated from the same small set of system prompts and templates. The test set was sampled from that same generation process, which is why the in-distribution metrics came out perfect (1.0 accuracy, F1, AUC-ROC). That's not generalization — that's the model recognizing the fingerprint of my own prompts. On truly out-of-distribution text, those numbers wouldn't hold.

**The problem is harder than the scope.** AI detection is adversarial: the moment you publish a classifier, anyone generating fakes can probe it and engineer around it. Even teams doing this full-time (GPTZero, Originality.ai) have significant false positive rates on real human writing. A DistilBERT trained on ~5k self-generated examples was never going to close that gap. The right dataset would need to be large, diverse, and sourced from generators I had no hand in prompting — which is a much bigger project.

**What fixing it would actually take:**
- A large, externally sourced AI detection dataset (existing NLP papers have them)
- Evaluation on held-out data generated with zero-shot prompts I didn't design
- Accepting real false positives and building UX around uncertainty rather than binary flags

That's a different project, and I'm not excited enough about AI detection specifically to build it.

## What I learned

This was still a useful build. Things that came out of it:

- Writing a raw PyTorch training loop end-to-end: custom `Dataset`, `DataLoader`, `AdamW` + linear warmup scheduler, gradient clipping, early stopping, checkpointing by validation F1
- Using `DistilBertModel` directly (not `DistilBertForSequenceClassification`) and attaching a custom classifier head
- Handling class imbalance with `BCEWithLogitsLoss(pos_weight=...)` rather than oversampling
- Building a data generation pipeline with checkpointing, retry logic, and star rating distributions that mirror real Yelp data
- FastAPI: lifespan model loading, Pydantic v2 validators, singleton inference wrapper
- Docker multi-platform builds (`--platform linux/amd64`) for ECS Fargate on Apple Silicon
- Chrome extension Manifest V3: content scripts, background service worker, `chrome.storage.sync`, MutationObserver for infinite scroll
- Deploying a containerized ML model to ECS Fargate with ECR

## Known code issues (documented for reference)

- **`predict_batch` is not batched** — loops `predict()` one at a time instead of running a single batched forward pass
- **`DistilBertClassifier` is copy-pasted** across `train.py`, `api/model.py`, and `stress_test.py` — should be one shared module
- **MutationObserver leak in the extension** — `observeNewReviews` creates a new observer on every call without storing the reference or calling `.disconnect()`
- **No auth or rate limiting on the API** — publicly exposed HTTP endpoint with `allow_origins=["*"]`
- **HTML template opened with a relative path** — `open("api/templates/index.html")` breaks if uvicorn isn't started from the project root
- **`MAX_LEN=256` may silently truncate** — subword tokenization means a 200-word review can exceed 256 tokens; DistilBERT supports up to 512
- **Adjusted rating is computed from visible reviews only** — paginated platforms don't load the full review corpus, so the recalculated rating is from a nonrepresentative sample

## Stack

- Python, PyTorch, HuggingFace Transformers
- FastAPI, Pydantic, Uvicorn
- Docker, AWS ECR, ECS Fargate
- Chrome Extension (Manifest V3)
- Anthropic API (Claude Haiku), OpenAI API (GPT-4o)
