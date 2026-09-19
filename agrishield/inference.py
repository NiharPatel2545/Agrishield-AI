from __future__ import annotations

from datetime import date, timedelta

from agrishield.config import EXAMPLE_SITE, FEATURE_COLUMNS
from agrishield import gee, weather
from agrishield.geography import approximate_continent

# Straight from continent_holdout_check() on the final tuned XGBoost model
# (see /areas/agrishield notes) -- acidic-class recall when that continent
# was held ENTIRELY out of training. This is an honest "how much did the
# model actually validate against something like your region" signal, not
# a made-up confidence score. Update this dict if you rerun the holdout
# check after any retrain -- it will go stale otherwise.
_CONTINENT_HOLDOUT_RECALL = {
    "South America": 0.789,
    "Oceania": 0.673,
    "Europe": 0.652,
    "Asia": 0.569,
    "Africa": 0.564,
    "Northern America": 0.497,
}


def _confidence_for(continent: str | None) -> dict:
    if continent is None or continent not in _CONTINENT_HOLDOUT_RECALL:
        return {
            "continent": continent,
            "tier": "unknown",
            "note": "Coordinates fall outside the regions this model has been validated against.",
        }
    recall = _CONTINENT_HOLDOUT_RECALL[continent]
    if recall >= 0.65:
        tier = "higher"
    elif recall >= 0.55:
        tier = "moderate"
    else:
        tier = "lower"
    return {
        "continent": continent,
        "tier": tier,
        "held_out_recall": recall,
        "note": (
            f"When {continent} was fully excluded from training and tested cold, "
            f"the model still caught {recall:.0%} of real acidic soil there. "
            "Treat this as a regional reliability signal, not the model's own confidence."
        ),
    }


def live_model_inputs(latitude: float, longitude: float) -> dict:
    """Flat dict with exactly the keys in FEATURE_COLUMNS -- this is what
    gets handed to model.predict_proba(). Anything this function can't
    fill comes back as None, and the model's imputer (median-fill, fit at
    training time) covers the gap -- same behaviour as a missing lab value.

    Copies EVERY key present in each GEE response, not a hand-picked
    subset -- a prior version hardcoded which keys to copy from each
    source (six of nine Sentinel bands, four of five static-soil keys,
    two of four WorldClim keys) and silently dropped the rest to None on
    every single live call, regardless of what GEE actually returned.
    Copying by iterating FEATURE_COLUMNS itself means adding a column to
    config.py is enough -- no second list to remember to update here.
    """
    gee.initialize()
    today = date.today()
    # 365 days, not 30. Training builds one full-calendar-year cloud-free
    # median composite per sample_year (climate.attach_sentinel_batched) --
    # a 30-day window here was a real train/serve mismatch: the model was
    # fit on year-long medians but served on a month-long one, which is a
    # different aggregation of a different-sized cloud/season mix. A
    # rolling 365-day window is the closest live equivalent to "a year's
    # median" without needing a fixed calendar year at request time.
    window_start = (today - timedelta(days=365)).isoformat()
    window_end = today.isoformat()

    out = {col: None for col in FEATURE_COLUMNS}

    try:
        bands = gee.sentinel2_features(longitude, latitude, window_start, window_end)
        for col, val in bands.items():
            if col in out:
                out[col] = val
    except Exception as exc:
        out["_sentinel_error"] = str(exc)

    try:
        soil = gee.static_soil_at_point(longitude, latitude)
        for col, val in soil.items():
            if col in out:
                out[col] = val
    except Exception as exc:
        out["_soil_error"] = str(exc)

    try:
        clim = gee.worldclim_at_point(longitude, latitude)
        for col, val in clim.items():
            if col in out:
                out[col] = val
    except Exception as exc:
        out["_climate_error"] = str(exc)

    return out


def live_features(latitude: float, longitude: float) -> dict:
    """Model inputs PLUS current weather AND a regional confidence signal --
    for the human-facing report, not for the model itself. current_weather
    is deliberately excluded from FEATURE_COLUMNS: it's today's snapshot,
    not something present (in this form) for any 2015/2018 training row.
    """
    continent = approximate_continent(latitude, longitude)
    out = {
        "latitude": latitude,
        "longitude": longitude,
        "model_inputs": live_model_inputs(latitude, longitude),
        "confidence": _confidence_for(continent),
    }
    try:
        out["weather_now"] = weather.current_weather(latitude, longitude)
    except Exception as exc:
        out["weather_error"] = str(exc)
    return out


def example_site_features() -> dict:
    return live_features(EXAMPLE_SITE["latitude"], EXAMPLE_SITE["longitude"])