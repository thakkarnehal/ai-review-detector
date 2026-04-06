"""
main.py
-------
FastAPI app for the AI-generated review detector.

Endpoints:
    GET  /health          — liveness check
    POST /detect          — classify a single review
    POST /adjust-rating   — recalculate a restaurant's rating after stripping fakes

Run locally:
    uvicorn api.main:app --reload --port 8080
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from api.model import get_classifier


# ── Lifespan: load model once on startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()   # warm up — raises on startup if weights are missing
    yield


app = FastAPI(
    title="AI Review Detector",
    description="Detects AI-generated restaurant reviews using DistilBERT.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in prod if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    review_text: str = Field(..., min_length=5, max_length=5000)


class DetectResponse(BaseModel):
    label:      str
    confidence: float
    flagged:    bool


class ReviewIn(BaseModel):
    text:        str = Field(..., min_length=1)
    star_rating: int = Field(..., ge=1, le=5)


class AdjustRatingRequest(BaseModel):
    reviews: list[ReviewIn] = Field(..., min_length=1)

    @field_validator("reviews")
    @classmethod
    def at_least_one(cls, v):
        if not v:
            raise ValueError("reviews list must not be empty")
        return v


class ReviewResult(BaseModel):
    text:        str
    star_rating: int
    label:       str
    confidence:  float
    flagged:     bool


class AdjustRatingResponse(BaseModel):
    original_rating: Optional[float]
    adjusted_rating: Optional[float]   # None if all reviews are flagged
    total_reviews:   int
    flagged_count:   int
    reviews:         list[ReviewResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    clf = get_classifier()
    return {"status": "ok", "model_loaded": clf is not None}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    clf    = get_classifier()
    result = clf.predict(req.review_text)
    return DetectResponse(**result)


@app.post("/adjust-rating", response_model=AdjustRatingResponse)
def adjust_rating(req: AdjustRatingRequest):
    clf = get_classifier()

    results: list[ReviewResult] = []
    for r in req.reviews:
        pred = clf.predict(r.text)
        results.append(ReviewResult(
            text        = r.text,
            star_rating = r.star_rating,
            **pred,
        ))

    total          = len(results)
    flagged_count  = sum(1 for r in results if r.flagged)
    all_ratings    = [r.star_rating for r in results]
    real_ratings   = [r.star_rating for r in results if not r.flagged]

    original_rating = round(sum(all_ratings) / total, 2)
    adjusted_rating = round(sum(real_ratings) / len(real_ratings), 2) if real_ratings else None

    return AdjustRatingResponse(
        original_rating = original_rating,
        adjusted_rating = adjusted_rating,
        total_reviews   = total,
        flagged_count   = flagged_count,
        reviews         = results,
    )


# ── HTML frontend ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def frontend():
    return open("api/templates/index.html").read()
