from __future__ import annotations

from agrishield.config import EXAMPLE_SITE
from agrishield import gee, weather


def live_features(latitude: float, longitude: float) -> dict:
    """Satellite + weather for a new location. Not training labels."""
    out = {"latitude": latitude, "longitude": longitude}
    try:
        gee.initialize()
        out["sentinel"] = gee.sentinel2_ndvi(longitude, latitude)
        out["worldclim"] = gee.worldclim_at_point(longitude, latitude)
    except Exception as exc:
        out["gee_error"] = str(exc)
    try:
        out["weather"] = weather.current_weather(latitude, longitude)
    except Exception as exc:
        out["weather_error"] = str(exc)
    return out


def example_site_features() -> dict:
    return live_features(EXAMPLE_SITE["latitude"], EXAMPLE_SITE["longitude"])
