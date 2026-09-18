from __future__ import annotations

from pathlib import Path

import pandas as pd

from agrishield.config import PROJECT_ROOT

CROP_CSV_PATH = PROJECT_ROOT / "data" / "crop_requirements.csv"


def _load_crop_table(path: Path = CROP_CSV_PATH) -> pd.DataFrame:
    """Loaded fresh each call -- this is a small reference table (a few KB),
    not the training data, so re-reading it costs nothing and means editing
    the CSV takes effect immediately with no restart needed."""
    return pd.read_csv(path)


def farmer_report(lat: float, lon: float, model, live_inputs: dict, prediction: dict,
                   weather: dict | None = None, crop: str | None = None) -> dict:
    """Turn a raw model prediction into a farmer-facing summary.

    prediction is whatever agrishield.model.predict_proba() returned.
    live_inputs is whatever agrishield.inference.live_model_inputs() returned.
    crop is optional -- e.g. "potato", "blueberry" -- matched against
    data/crop_requirements.csv, NOT used to train the model itself.
    """
    acidic_prob = prediction["probability"].get(1, 0.0)
    predicted_acidic = prediction["predicted"] == 1

    if acidic_prob >= 0.65:
        verdict = "acidic"
        headline = "Soil likely acidic"
        action = "Consider a lime treatment and a follow-up soil test before planting."
        color = "#EF4444"
    elif acidic_prob >= 0.40:
        verdict = "uncertain"
        headline = "Uncertain -- borderline result"
        action = "A physical soil test is recommended before making changes."
        color = "#F59E0B"
    else:
        verdict = "stable"
        headline = "Soil pH looks stable"
        action = "No action needed based on this scan."
        color = "#22C55E"

    crop_note = None
    if crop:
        crops = _load_crop_table()
        match = crops[crops["crop"].str.lower() == crop.lower()]
        if not match.empty:
            low, high = float(match.iloc[0]["ph_min"]), float(match.iloc[0]["ph_max"])
            if predicted_acidic and high <= 5.8:
                verdict = "suitable"
                headline = f"Acidic soil -- but that suits {crop}"
                action = f"{crop.title()} tolerates pH {low}-{high}; no lime treatment needed."
                color = "#22C55E"
            crop_note = f"{crop.title()} typically prefers pH {low}-{high}."

    used_imputed_satellite = live_inputs.get("B2") is None

    return {
        "location": {"latitude": lat, "longitude": lon},
        "headline": headline,
        "verdict": verdict,
        "confidence_pct": round(acidic_prob * 100, 1) if predicted_acidic else round((1 - acidic_prob) * 100, 1),
        "recommended_action": action,
        "crop_note": crop_note,
        "color": color,
        "context": {
            "ndvi": live_inputs.get("ndvi"),
            "avg_temp_c": live_inputs.get("tmean_c"),
            "avg_annual_rainfall_mm": live_inputs.get("precip_mm"),
            "current_weather": weather.get("description") if weather else None,
        },
        "caveats": (
            ["Satellite data unavailable for this exact spot/date -- result relies more on climate and soil-texture maps."]
            if used_imputed_satellite else []
        ),
    }


def suggest_crops(prediction: dict, top_n: int = 5) -> list[str]:
    """Reverse direction: no crop given -- suggest ones that suit THIS soil."""
    acidic_prob = prediction["probability"].get(1, 0.0)
    is_acidic = acidic_prob >= 0.5
    crops = _load_crop_table()
    if is_acidic:
        matches = crops[crops["ph_max"] <= 5.8]
    else:
        matches = crops[crops["ph_min"] >= 5.8]
    return matches["crop"].str.title().tolist()[:top_n]