from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LUCAS_2018 = PROJECT_ROOT / "LUCAS 2018" / "LUCAS-SOIL-2018-v2" / "LUCAS-SOIL-2018.csv"
LUCAS_2015 = PROJECT_ROOT / "LUCAS 2015" / "LUCAS_Topsoil_2015_20200323.csv"
LUCAS_BD = PROJECT_ROOT / "LUCAS 2018" / "LUCAS-SOIL-2018-v2" / "BulkDensity_2018_final-2.csv"
WOSIS_DIR = PROJECT_ROOT / "WoSIS"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_CSV = DATA_DIR / "agrishield_training.csv"
MODEL_PATH = MODELS_DIR / "soil_risk_rf.joblib"

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "valued-aquifer-507001-v2")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

FEATURE_COLUMNS = [
    "oc_gkg",
    "n_gkg",
    "p_mgkg",
    "k_mgkg",
    "ec",
    "caco3",
    "clay_pct",
    "sand_pct",
    "silt_pct",
    "coarse_pct",
    "elevation_m",
    "bd_0_20",
    "cec_ph7",
    "totc_gkg",
    "tmean_c",
    "precip_mm",
]
TARGET_COLUMN = "acidic"

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
