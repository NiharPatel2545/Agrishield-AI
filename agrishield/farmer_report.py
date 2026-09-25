from __future__ import annotations

from pathlib import Path

import pandas as pd

from agrishield.config import PROJECT_ROOT

CROP_CSV_PATH = PROJECT_ROOT / "data" / "crop_requirements.csv"


def _load_crop_table(path: Path = CROP_CSV_PATH) -> pd.DataFrame | None:
    """Loaded fresh each call -- this is a small reference table (a few KB),
    not the training data, so re-reading it costs nothing and means editing
    the CSV takes effect immediately with no restart needed.

    Returns None (not an exception) if the file doesn't exist yet. Without
    this, every /scan call with no crop given still hits suggest_crops()
    below, which unconditionally read this file -- a 100% crash rate on
    the live API until a real crop_requirements.csv is added."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def farmer_report(lat: float, lon: float, model, live_inputs: dict, prediction: dict,
                   weather: dict | None = None, crop: str | None = None,
                   predicted_oc: float | None = None) -> dict:
    """Turn a raw model prediction into a farmer-facing summary.

    prediction is whatever agrishield.model.predict_proba() returned.
    live_inputs is whatever agrishield.inference.live_model_inputs() returned.
    crop is optional -- e.g. "potato", "blueberry" -- matched against
    data/crop_requirements.csv, NOT used to train the model itself.
    predicted_oc is optional -- whatever agrishield.model.predict_oc() returned
    from the SEPARATE organic-carbon regression model. None if that model
    hasn't been wired in yet (keeps this function working either way).
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
        if crops is None:
            crop_note = f"No crop_requirements.csv yet -- can't check {crop}'s pH tolerance."
        else:
            match = crops[crops["crop"].str.lower() == crop.lower()]
        if crops is not None and not match.empty:
            low, high = float(match.iloc[0]["ph_min"]), float(match.iloc[0]["ph_max"])
            if predicted_acidic and high <= 5.8:
                verdict = "suitable"
                headline = f"Acidic soil -- but that suits {crop}"
                action = f"{crop.title()} tolerates pH {low}-{high}; no lime treatment needed."
                color = "#22C55E"
            crop_note = f"{crop.title()} typically prefers pH {low}-{high}."

    # Real quartiles from the actual training data (see /areas/agrishield
    # notes): 25th pct ~10.3, 75th pct ~35.7 g/kg. Not round guesses.
    oc_note = None
    if predicted_oc is not None:
        if predicted_oc < 10.3:
            oc_note = ("Low organic carbon -- soil may be depleted; consider compost/organic "
                        "matter before adding synthetic fertilizer.")
        elif predicted_oc > 35.7:
            oc_note = "Organic carbon looks healthy -- likely less need for additional fertilizer."
        else:
            oc_note = "Organic carbon in a typical mid-range -- no strong signal either way."

    used_imputed_satellite = live_inputs.get("B2") is None

    return {
        "location": {"latitude": lat, "longitude": lon},
        "headline": headline,
        "verdict": verdict,
        "confidence_pct": round(acidic_prob * 100, 1) if predicted_acidic else round((1 - acidic_prob) * 100, 1),
        "recommended_action": action,
        "crop_note": crop_note,
        "oc_note": oc_note,
        "predicted_oc_gkg": round(predicted_oc, 1) if predicted_oc is not None else None,
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


# A small set of crops most people will actually recognize -- shown first
# whenever they match, so a demo doesn't surface "Abbo Rubber Tree" or
# "Muraina Grass" just because they happen to sort alphabetically ahead
# of wheat or tomato. This doesn't change WHICH crops are considered a
# match (still real EcoCrop pH data) -- only which matching ones get
# shown first. Extend this list any time; it's just a display priority.
_COMMON_CROPS = {
    "wheat", "rice", "maize", "corn", "barley", "oats", "potato", "tomato",
    "onion", "garlic", "carrot", "cabbage", "lettuce", "spinach", "pea",
    "bean", "soybean", "lentil", "chickpea", "groundnut", "peanut",
    "sunflower", "cotton", "sugarcane", "banana", "mango", "apple",
    "orange", "grape", "coffee", "cocoa", "cassava", "sweet potato", "yam",
    # Previously nothing here had ph_max <= 5.8, so every acidic-soil result
    # fell through to obscure alphabetical species (Abbo Rubber Tree, Muraina
    # Grass) instead of anything recognizable. These are real, common,
    # acidic-tolerant crops confirmed present in crop_requirements.csv.
    "avocado", "cranberry", "blueberry", "high-bush blueberry",
    "low-bush blueberry", "yam bean",
}


def suggest_crops(prediction: dict, top_n: int = 5) -> list[str]:
    """Reverse direction: no crop given -- suggest ones that suit THIS soil.
    Returns [] (not an exception) if crop_requirements.csv doesn't exist yet --
    this is what api.py's /scan calls on every crop-less request, so a hard
    crash here means every such request 500s."""
    crops = _load_crop_table()
    if crops is None:
        return []
    acidic_prob = prediction["probability"].get(1, 0.0)
    is_acidic = acidic_prob >= 0.5
    if is_acidic:
        matches = crops[crops["ph_max"] <= 5.8]
    else:
        matches = crops[crops["ph_min"] >= 5.8]

    names = matches["crop"].tolist()
    common = [n for n in names if n.lower() in _COMMON_CROPS]
    rest = [n for n in names if n.lower() not in _COMMON_CROPS]
    ordered = common + rest  # common matches first, alphabetical fallback fills the rest
    return [n.title() for n in ordered[:top_n]]