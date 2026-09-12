from __future__ import annotations

from datetime import date, timedelta

from agrishield.config import EXAMPLE_SITE, FEATURE_COLUMNS
from agrishield import gee, weather


def live_model_inputs(latitude: float, longitude: float) -> dict:
    """Flat dict with exactly the keys in FEATURE_COLUMNS -- this is what
    gets handed to model.predict_proba(). Anything this function can't
    fill comes back as None, and the model's imputer (median-fill, fit at
    training time) covers the gap -- same behaviour as a missing lab value.
    """
    gee.initialize()
    today = date.today()
    window_start = (today - timedelta(days=30)).isoformat()
    window_end = today.isoformat()

    out = {col: None for col in FEATURE_COLUMNS}
    try:
        bands = gee.sentinel2_features(longitude, latitude, window_start, window_end)
        for col in ["B2", "B3", "B4", "B8", "B11", "ndvi"]:
            if col in bands:
                out[col] = bands[col]
    except Exception as exc:
        out["_sentinel_error"] = str(exc)

    try:
        soil = gee.static_soil_at_point(longitude, latitude)
        for col in ["clay_pct", "sand_pct", "silt_pct", "elevation_m"]:
            if col in soil:
                out[col] = soil[col]
    except Exception as exc:
        out["_soil_error"] = str(exc)

    try:
        clim = gee.worldclim_at_point(longitude, latitude)
        out["tmean_c"] = clim.get("tmean_c")
        out["precip_mm"] = clim.get("precip_mm")
    except Exception as exc:
        out["_climate_error"] = str(exc)

    return out


def live_features(latitude: float, longitude: float) -> dict:
    """Model inputs PLUS current weather -- for the human-facing report,
    not for the model itself. current_weather is deliberately excluded
    from FEATURE_COLUMNS: it's today's snapshot, not something present
    (in this form) for any 2015/2018 training row.
    """
    out = {
        "latitude": latitude,
        "longitude": longitude,
        "model_inputs": live_model_inputs(latitude, longitude),
    }
    try:
        out["weather_now"] = weather.current_weather(latitude, longitude)
    except Exception as exc:
        out["weather_error"] = str(exc)
    return out


def example_site_features() -> dict:
    return live_features(EXAMPLE_SITE["latitude"], EXAMPLE_SITE["longitude"])