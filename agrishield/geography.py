"""Approximate continent lookup from raw lat/lon, with no external service call.

This is a rough bounding-box classifier, not a real reverse-geocoder --
good enough to pick a confidence tier for the API response, NOT good
enough for anything where continent boundaries matter precisely. The
Mediterranean is the known weak spot: the real Europe/Africa boundary
there isn't axis-aligned, so no rectangular box gets it fully right.
Africa is checked before Europe so North African mainland coastal
cities (Algiers, Tunis, Tripoli) classify correctly -- the trade-off is
that small Mediterranean islands at similar latitude (Malta, Crete,
southern Sicily) will now misclassify as Africa instead. If this ever
needs to be exact, swap in a real reverse-geocoding library
(e.g. reverse_geocoder) instead of tightening these boxes further.
"""

from __future__ import annotations

# (min_lat, max_lat, min_lon, max_lon) -- checked in this order, first
# match wins, so put narrower/more specific boxes before broad ones.
# Africa's lat_max is capped at 37.3 (Ras ben Sakka, Tunisia -- the actual
# northernmost point of the African mainland), not a rounder-looking 38.
# That distinction matters: at lat_max=38 this box swallowed Athens
# (37.98N) and Sicily (~38.1N) as Africa, which is a much bigger miss
# than the island edge cases below. At 37.3 it correctly keeps Algiers/
# Tunis/Tripoli/Cairo as Africa while letting Athens/Sicily/Rome/Madrid
# fall through to Europe correctly. Small Mediterranean islands at
# similar latitude to the Tunisian/Libyan coast (Malta ~35.9N, Crete
# ~35N) still land in Africa -- a real remaining edge case, but a much
# smaller population than Athens.
_BOXES: list[tuple[str, float, float, float, float]] = [
    ("Antarctica", -90, -60, -180, 180),
    ("Africa", -35, 37.3, -20, 52),
    ("Europe", 34, 72, -25, 45),
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
