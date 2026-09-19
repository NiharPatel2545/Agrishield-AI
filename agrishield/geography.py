"""Approximate continent lookup from raw lat/lon, with no external service call.

This is a rough bounding-box classifier, not a real reverse-geocoder --
good enough to pick a confidence tier for the API response, NOT good
enough for anything where continent boundaries matter precisely (e.g. a
coordinate right on a Europe/Asia border might come back wrong). If this
ever needs to be exact, swap in a real reverse-geocoding library
(e.g. reverse_geocoder) instead of tightening these boxes further.
"""

from __future__ import annotations

# (min_lat, max_lat, min_lon, max_lon) -- checked in this order, first
# match wins, so put narrower/more specific boxes before broad ones.
_BOXES: list[tuple[str, float, float, float, float]] = [
    ("Antarctica", -90, -60, -180, 180),
    ("Europe", 34, 72, -25, 45),
    ("Africa", -35, 38, -20, 52),
    ("Oceania", -50, 0, 110, 180),
    ("Asia", -10, 82, 45, 180),
    ("Northern America", 5, 85, -170, -50),
    ("South America", -60, 13, -85, -32),
]


def approximate_continent(latitude: float, longitude: float) -> str | None:
    for name, lat_min, lat_max, lon_min, lon_max in _BOXES:
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return name
    return None
