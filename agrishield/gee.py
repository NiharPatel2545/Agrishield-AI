from __future__ import annotations

import ee

from agrishield.config import GEE_PROJECT_ID

SENTINEL_BANDS = ["B2", "B3", "B4", "B8", "B11"]

# OpenLandMap surface (0 cm) texture fraction layers — global, static, free.
CLAY_IMG = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
SAND_IMG = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
SILT_IMG = "OpenLandMap/SOL/SOL_SILT-WFRACTION_USDA-3A1A1A_M/v02"
ELEVATION_IMG = "USGS/SRTMGL1_003"


def initialize() -> str:
    """Authenticate if needed, then initialise Earth Engine."""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=GEE_PROJECT_ID)
    return GEE_PROJECT_ID


def sentinel2_features(longitude: float, latitude: float, start: str, end: str) -> dict:
    """B2/B3/B4/B8/B11 + NDVI for one point, one date window.

    select() happens BEFORE median() — mixing images with different band
    orderings (a real Sentinel-2 quirk) makes median() throw
    'Expected a homogeneous image collection' otherwise.
    """
    point = ee.Geometry.Point([longitude, latitude])
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .select(SENTINEL_BANDS)
        .median()
    )
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    sample = image.addBands(ndvi).sample(point, 10).first()
    info = sample.getInfo() if sample else None
    return info.get("properties", {}) if info else {}


def worldclim_at_point(longitude: float, latitude: float) -> dict:
    """Long-term climate (bio01 = mean temp * 10, bio12 = annual precip mm)."""
    point = ee.Geometry.Point([longitude, latitude])
    image = ee.Image("WORLDCLIM/V1/BIO").select(["bio01", "bio12"])
    sample = image.sample(point, 1000).first()
    info = sample.getInfo() if sample else None
    props = info.get("properties", {}) if info else {}
    tmean = props.get("bio01")
    return {
        "tmean_c": None if tmean is None else tmean / 10.0,
        "precip_mm": props.get("bio12"),
    }


def static_soil_at_point(longitude: float, latitude: float) -> dict:
    """Clay/sand/silt % and elevation (m) from static global GEE layers.

    This must be the SAME source used for training rows (see
    climate.attach_static_soil) — never mix lab-measured texture with
    satellite-derived texture between train and inference.
    """
    point = ee.Geometry.Point([longitude, latitude])
    clay = ee.Image(CLAY_IMG).select("b0").rename("clay_pct")
    sand = ee.Image(SAND_IMG).select("b0").rename("sand_pct")
    silt = ee.Image(SILT_IMG).select("b0").rename("silt_pct")
    elev = ee.Image(ELEVATION_IMG).select("elevation").rename("elevation_m")
    stack = clay.addBands([sand, silt, elev])
    sample = stack.sample(point, 250).first()
    info = sample.getInfo() if sample else None
    return info.get("properties", {}) if info else {}