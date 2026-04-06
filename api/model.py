"""
model.py
--------
Model loading and inference logic. Imported by main.py on startup.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel

BASE_DIR       = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "best_model")
PRETRAINED     = "distilbert-base-uncased"
MAX_LEN        = 256
THRESHOLD      = 0.5


class DistilBertClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(PRETRAINED)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.dim, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls)).squeeze(-1)


class ReviewClassifier:
    """Singleton wrapper — load once on startup, reuse for every request."""

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ckpt = os.path.join(CHECKPOINT_DIR, "model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Model checkpoint not found at {ckpt}. "
                "Run scripts/train.py before starting the API."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        self.model     = DistilBertClassifier().to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Return label, confidence, and flagged flag for a single review."""
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logit = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
            prob = torch.sigmoid(logit).item()

        flagged = prob >= THRESHOLD
        return {
            "label":      "AI-generated" if flagged else "Real",
            "confidence": round(prob, 4),
            "flagged":    flagged,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run inference on a list of texts, one at a time (keeps memory low)."""
        return [self.predict(t) for t in texts]


# Module-level singleton — imported by main.py
classifier: ReviewClassifier | None = None


def get_classifier() -> ReviewClassifier:
    global classifier
    if classifier is None:
        classifier = ReviewClassifier()
    return classifier
