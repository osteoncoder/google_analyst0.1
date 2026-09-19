#!/usr/bin/env python3
"""
app.py — FastAPI backend for the APEX dashboard (ML sections 07-09).

Serves:
    /                      the static dashboard (index.html + assets)
    /api/health            model availability (no fake states)
    /api/metrics           combined evaluation metrics (section 09)
    /api/predict/rating    M1 inference — saved pipeline, NO retraining
    /api/predict/tier      M2 inference — primary model = without Reviews

If the model artifacts are missing (training never run), endpoints return
503 and the UI shows a clear "unavailable" state — never random/demo values.

Run:
    python app.py                 # http://0.0.0.0:8000
    # or: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
ART = ROOT / "ml" / "artifacts"

sys.path.insert(0, str(ROOT))
# Single source of truth: a hand-typed category is normalized exactly like the
# training categories were (trim, underscores/hyphens -> spaces, title case,
# documented typo fix) instead of being fed raw to the one-hot encoder.
from clean import clean_category  # noqa: E402

MODEL_SPECS = {
    "m1": ART / "m1_rating" / "pipeline.joblib",
    "m2": ART / "m2_tier_without_reviews" / "pipeline.joblib",  # primary
}

state: dict = {"m1": None, "m2": None, "m1_meta": None, "m2_meta": None}


def load_artifacts() -> None:
    for key, path in MODEL_SPECS.items():
        if path.exists():
            state[key] = joblib.load(path)
            meta_path = path.parent / "meta.json"
            if meta_path.exists():
                state[f"{key}_meta"] = json.loads(meta_path.read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="APEX ML API", lifespan=lifespan)


class RatingIn(BaseModel):
    category: str = Field(min_length=1, max_length=120)
    size_mb: float | None = Field(default=None, ge=0, le=1_000_000)
    price: float | None = Field(default=None, ge=0, le=100_000)
    reviews: int | None = Field(default=None, ge=0, le=10_000_000_000)

    @field_validator("reviews", mode="before")
    @classmethod
    def _round_review_count(cls, v):
        """Reviews is a count: a fractional value (e.g. 1200.5) is rounded, not
        rejected with a 422 — the model input is log1p(reviews)."""
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                raise ValueError("reviews must be a finite number")
            return int(round(v))
        return v


class TierIn(BaseModel):
    category: str = Field(min_length=1, max_length=120)
    size_mb: float | None = Field(default=None, ge=0, le=1_000_000)
    price: float | None = Field(default=None, ge=0, le=100_000)


def _assumptions(size_mb, price, reviews=None, has_reviews=False) -> list[str]:
    a = []
    if size_mb is None:
        a.append("size omitted → imputed to the training-set median")
    if price is None:
        a.append("price omitted → treated as free ($0)")
    # Only M1 has a reviews input; for M2 "no reviews field" is not an assumption.
    if has_reviews and reviews is None:
        a.append("reviews omitted → treated as 0")
    a.append("listed price is a price tag, not observed revenue")
    return a


def _build_frame(row: dict, meta: dict) -> pd.DataFrame:
    """Frame with EXACTLY the columns the saved preprocessor was fitted with."""
    return pd.DataFrame([row])[meta["features"]]


def _normalized_category(raw: str) -> str:
    """clean.py's category normalization — the same one applied to training data.

    Without this, a hand-typed category ("education") would not match the
    trained one ("Education") and the one-hot encoder would silently treat it
    as an unseen category, returning a different prediction for the same app.
    """
    cat = clean_category(raw)
    if cat is None or not cat.strip():
        raise HTTPException(status_code=422,
                            detail="category must contain at least one letter or digit")
    return cat.strip()


def _known_categories(pipe) -> set[str]:
    """Categories the fitted encoder actually saw during training."""
    try:
        enc = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        return {str(c) for c in enc.categories_[0]}
    except Exception:
        return set()


def _category_assumption(pipe, category: str) -> str | None:
    """Honest note when the (normalized) category was never seen in training."""
    known = _known_categories(pipe)
    if known and category not in known:
        return (f"category {category!r} was not seen in training → encoded as an unknown "
                f"category (the model knows {len(known)} categories)")
    return None


def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@app.get("/api/health")
def health():
    m1, m2 = state["m1"] is not None, state["m2"] is not None
    sample = bool(state["m2_meta"] and state["m2_meta"].get("dataset", {}).get("is_sample")) \
        or bool(state["m1_meta"] and state["m1_meta"].get("dataset", {}).get("is_sample"))
    return {
        "ok": True,
        "models_loaded": m1 and m2,
        "m1_rating": m1,
        "m2_tier_without_reviews": m2,
        "trained_on_sample_dataset": sample,
    }


@app.get("/api/metrics")
def metrics():
    p = ART / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=503, detail="Metrics not found — run: python train_models.py")
    return json.loads(p.read_text())


@app.post("/api/predict/rating")
def predict_rating(body: RatingIn):
    if state["m1"] is None:
        raise HTTPException(status_code=503, detail="M1 model not loaded — run: python train_models.py")
    meta = state["m1_meta"] or {}
    if not meta.get("features"):
        raise HTTPException(status_code=503, detail="M1 model metadata missing — re-run: python train_models.py")
    category = _normalized_category(body.category)
    row = {
        "category": category,
        "size_mb": body.size_mb if body.size_mb is not None else np.nan,
        "price": body.price if body.price is not None else 0.0,
        "price_is_positive": 1.0 if (body.price or 0.0) > 0 else 0.0,
        "reviews_log": float(np.log1p(body.reviews or 0)),
    }
    X = _build_frame(row, meta)
    raw_pred = float(state["m1"].predict(X)[0])
    # Play ratings are defined on [1, 5]; clip extrapolated predictions to the
    # target's own domain and say so when clipping was needed.
    pred = float(np.clip(raw_pred, 1.0, 5.0))
    assumptions = _assumptions(body.size_mb, body.price, body.reviews, has_reviews=True)
    if pred != raw_pred:
        assumptions.append(f"raw prediction {raw_pred:.3f} clipped to the rating domain [1, 5]")
    hint = _category_assumption(state["m1"], category)
    if hint:
        assumptions.append(hint)
    return {
        "predicted_rating": round(pred, 3),
        "category_used": category,
        "category_changed": body.category.strip() != category,
        "model": meta.get("model"),
        "test_metrics": _round(meta.get("test_metrics", {})),
        "n_test": meta.get("n_test"),
        "assumptions": assumptions,
        "warning": "Model estimate on a cross-sectional snapshot — not a pre-launch or future rating guarantee.",
    }


@app.post("/api/predict/tier")
def predict_tier(body: TierIn):
    if state["m2"] is None:
        raise HTTPException(status_code=503,
                            detail="M2 model not loaded — run: python train_models.py")
    meta = state["m2_meta"] or {}
    if not meta.get("features"):
        raise HTTPException(status_code=503, detail="M2 model metadata missing — re-run: python train_models.py")
    category = _normalized_category(body.category)
    row = {
        "category": category,
        "size_mb": body.size_mb if body.size_mb is not None else np.nan,
        "price": body.price if body.price is not None else 0.0,
        "price_is_positive": 1.0 if (body.price or 0.0) > 0 else 0.0,
    }
    X = _build_frame(row, meta)
    pipe = state["m2"]
    proba = pipe.predict_proba(X)[0]
    labels = list(pipe.classes_)
    probs = {str(l): float(p) for l, p in zip(labels, proba)}
    pred_tier = str(labels[int(np.argmax(proba))])
    tm = meta.get("test_metrics", {})
    assumptions = _assumptions(body.size_mb, body.price)
    hint = _category_assumption(pipe, category)
    if hint:
        assumptions.append(hint)
    return {
        "predicted_tier": pred_tier,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "tier_order": meta.get("labels", TIER_ORDER_FALLBACK),
        "category_used": category,
        "category_changed": body.category.strip() != category,
        "model": meta.get("model"),
        "test_metrics": _round({k: v for k, v in tm.items() if isinstance(v, float)}),
        "assumptions": assumptions,
        "warning": "Probabilities are model estimates, not guarantees. "
                   "This model deliberately excludes Reviews (target proxy).",
    }


TIER_ORDER_FALLBACK = ["Under 10K", "10K-1M", "1M-100M", "100M+"]


class _SafeStatic(StaticFiles):
    """Static file serving that never exposes dot-directories (.git, .venv, …)."""

    async def get_response(self, path: str, scope) -> "object":
        parts = Path(path).parts
        if any(p.startswith(".") for p in parts if p not in (".",)):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)


app.mount("/", _SafeStatic(directory=ROOT, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    HOST, PORT = "0.0.0.0", 8000
    # uvicorn logs the BIND address (0.0.0.0 = every interface), which a browser
    # cannot open — on Windows it fails with ERR_ADDRESS_NOT_AVAILABLE, and VS
    # Code makes it a clickable link. Print the address that actually works.
    print(f"\n  APEX dashboard    ->  http://localhost:{PORT}")
    print(f"  API health check  ->  http://localhost:{PORT}/api/health")
    print("  (the 0.0.0.0 in uvicorn's log is the bind address, not a URL —\n"
          "   open localhost or 127.0.0.1 in your browser)\n")
    sys.stdout.flush()          # print before uvicorn's own log line
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
