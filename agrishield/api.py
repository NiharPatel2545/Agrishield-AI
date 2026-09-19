"""FastAPI service: the one thing everything else (website, crop suggestions,
confidence indicator) depends on existing. Run locally with:
    uvicorn api:app --reload
Then hit http://127.0.0.1:8000/docs for an interactive test page for free
(FastAPI auto-generates this -- no extra work needed).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel

from agrishield.model import load_model, predict_proba
from agrishield.inference import live_features, live_model_inputs
from agrishield.farmer_report import farmer_report, suggest_crops
from agrishield import weather as weather_module

app = FastAPI(title="Agrishield API", version="0.1.0")

# Loaded ONCE at startup, not per-request -- loading a joblib from disk on
# every API call would be needlessly slow and is the classic mistake here.
_model = load_model()


# ---- basic API-key + rate limiting -------------------------------------
# Every request below triggers a live Google Earth Engine call. GEE quota
# is not infinite (see /areas/agrishield notes -- noncommercial tier has a
# monthly EECU-hour budget). An unprotected public endpoint can get hit by
# bots/scrapers and burn real quota with zero real farmers involved. This
# is a deliberately SIMPLE in-memory limiter -- fine for one server
# instance / a prototype, NOT sufficient if you ever run multiple server
# processes (each would keep its own separate counts). Swap for a shared
# store (e.g. Redis) if you scale beyond one instance.
API_KEY = os.getenv("AGRISHIELD_API_KEY", "")
_RATE_LIMIT_PER_MINUTE = 20
_request_log: dict[str, list[float]] = defaultdict(list)


def _check_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _check_rate_limit(key: str) -> None:
    now = time.time()
    recent = [t for t in _request_log[key] if now - t < 60]
    if len(recent) >= _RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded -- try again in a minute.")
    recent.append(now)
    _request_log[key] = recent


class ScanResponse(BaseModel):
    location: dict
    headline: str
    verdict: str
    confidence_pct: float
    recommended_action: str
    crop_note: str | None
    color: str
    context: dict
    caveats: list[str]
    regional_confidence: dict
    suggested_crops: list[str] | None = None


@app.get("/health")
def health():
    """Cheap endpoint with no GEE call -- use this for uptime monitoring,
    not /scan, so you're not burning GEE quota just to check the server
    is alive."""
    return {"status": "ok"}


@app.get("/scan", response_model=ScanResponse)
def scan(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    crop: str | None = Query(None, description="Optional crop name to check fit against, e.g. 'rice'"),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)
    _check_rate_limit(x_api_key or "anonymous")

    try:
        full = live_features(lat, lon)
    except Exception as exc:
        # Never let a farmer see a raw Python traceback -- this is the
        # error-handling item from the roadmap. GEE/weather being briefly
        # unreachable should read as "try again shortly", not a stack trace.
        raise HTTPException(
            status_code=503,
            detail="Soil data service is temporarily unavailable. Please try again shortly.",
        ) from exc

    prediction = predict_proba(_model, full["model_inputs"])
    weather_now = full.get("weather_now")

    report = farmer_report(
        lat=lat, lon=lon, model=_model,
        live_inputs=full["model_inputs"], prediction=prediction,
        weather=weather_now, crop=crop,
    )
    report["regional_confidence"] = full["confidence"]

    if crop is None:
        report["suggested_crops"] = suggest_crops(prediction)

    return report


DISCLAIMER = (
    "Agrishield provides soil-risk guidance based on satellite and climate data. "
    "It is not a replacement for a professional laboratory soil test, especially "
    "before major decisions like lime treatment or crop changes."
)


@app.get("/")
def root():
    return {"service": "Agrishield API", "disclaimer": DISCLAIMER, "docs": "/docs"}
