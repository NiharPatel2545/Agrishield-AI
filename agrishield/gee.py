from __future__ import annotations

import ee

from agrishield.config import GEE_PROJECT_ID


def initialize() -> str:
    """Authenticate if needed, then initialise Earth Engine."""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=GEE_PROJECT_ID)
    return GEE_PROJECT_ID


def sentinel2_ndvi(longitude: float, latitude: float, start: str = "2025-08-01", end: str = "2025-09-01") -> dict:
    point = ee.Geometry.Point([longitude, latitude])
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .median()
    )
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    sample = ndvi.addBands(image.select(["B4", "B8"])).sample(point, 10).first()
    info = sample.getInfo()
    return info.get("properties", {}) if info else {}


def worldclim_at_point(longitude: float, latitude: float) -> dict:
    """Long-term climate (bio01 = mean temp * 10, bio12 = annual precip mm)."""
    point = ee.Geometry.Point([longitude, latitude])
    image = ee.Image("WORLDCLIM/V1/BIO").select(["bio01", "bio12"])
    sample = image.sample(point, 1000).first().getInfo()
    props = sample.get("properties", {}) if sample else {}
    tmean = props.get("bio01")
    return {
        "tmean_c": None if tmean is None else tmean / 10.0,
        "precip_mm": props.get("bio12"),
    }
