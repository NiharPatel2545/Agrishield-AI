from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LUCAS_2018 = PROJECT_ROOT / "LUCAS 2018" / "LUCAS-SOIL-2018-v2" / "LUCAS-SOIL-2018.csv"
LUCAS_2015 = PROJECT_ROOT / "LUCAS 2015" / "LUCAS_Topsoil_2015_20200323.csv"
LUCAS_2009 = PROJECT_ROOT / "LUCAS 2009" / "LUCAS_TOPSOIL_v1.xlsx"
LUCAS_BD = PROJECT_ROOT / "LUCAS 2018" / "LUCAS-SOIL-2018-v2" / "BulkDensity_2018_final-2.csv"
WOSIS_DIR = PROJECT_ROOT / "WoSIS"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_CSV = DATA_DIR / "agrishield_training.csv"
MODEL_PATH = MODELS_DIR / "soil_risk_rf.joblib"

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "valued-aquifer-507001-v2")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Every one of these must be obtainable from inference.live_features() for a
# brand-new coordinate with NO lab test. This is the actual hard constraint the
# model has to satisfy. Do not add a lab/survey-only column here.
FEATURE_COLUMNS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12",  # Sentinel-2 bands, live
    "ndvi",                              # derived from B8/B4, live
    "elevation_m", "slope_deg",          # SRTM + derived terrain, live
    "clay_pct", "sand_pct", "silt_pct",  # OpenLandMap static maps, live (silt derived)
    "tmean_c", "temp_seasonality",       # WorldClim climatology, live
    "precip_mm", "precip_seasonality",   # WorldClim climatology, live
]
# Lab-only columns (oc_gkg, n_gkg, p_mgkg, k_mgkg, ec, caco3, bd_0_20,
# cec_ph7, totc_gkg) may only ever be a TARGET, never a feature — no live
# API can ever produce them for a new farmer coordinate.
TARGET_COLUMN = "acidic"  # derived from ph_h2o; swap to "oc_gkg" for a regression target

EXAMPLE_SITE = {
    "name": "Clayton, Victoria",
    "latitude": -37.91,
    "longitude": 145.13,
}

WOSIS_SKIP = {"sites", "profiles", "layers", "observations"}

WOSIS_CANONICAL = {
    "orgc": "oc_gkg",
    "phaq": "ph_h2o",
    "phca": "ph_cacl2",
    "clay": "clay_pct",
    "sand": "sand_pct",
    "silt": "silt_pct",
    "nitkjd": "n_gkg",
    "totc": "totc_gkg",
    "cecph7": "cec_ph7",
    "elco25": "ec",
    "tceq": "caco3",
    "bdfiod": "bd_0_20",
}