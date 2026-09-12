from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agrishield.config import (
    DATA_DIR,
    LUCAS_2009,
    LUCAS_2015,
    LUCAS_2018,
    LUCAS_BD,
    TRAINING_CSV,
    WOSIS_CANONICAL,
    WOSIS_DIR,
    WOSIS_SKIP,
)

WOSIS_COLS = [
    "profile_id",
    "upper_depth",
    "value_avg",
    "country_name",
    "longitude",
    "latitude",
    "continent",
    "date",
]


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"< LOD": np.nan, "<LOD": np.nan, "nan": np.nan, "": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _numeric_cols(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = to_numeric(df[col])


def load_lucas_2018() -> pd.DataFrame:
    raw = pd.read_csv(LUCAS_2018, low_memory=False)
    _numeric_cols(
        raw,
        [
            "TH_LAT",
            "TH_LONG",
            "Elev",
            "pH_H2O",
            "pH_CaCl2",
            "EC",
            "OC",
            "CaCO3",
            "P",
            "N",
            "K",
            "OC (20-30 cm)",
            "CaCO3 (20-30 cm)",
            "Ox_Al",
            "Ox_Fe",
        ],
    )
    if LUCAS_BD.exists():
        bd = pd.read_csv(LUCAS_BD)
        bd = bd.rename(columns={"POINT_ID": "POINTID", "BD 0-20": "bd_0_20"})
        if "bd_0_20" in bd.columns:
            raw = raw.merge(bd[["POINTID", "bd_0_20"]], on="POINTID", how="left")

    tex = pd.read_csv(LUCAS_2015, usecols=["Point_ID", "Clay", "Sand", "Silt", "Coarse"], low_memory=False)
    _numeric_cols(tex, ["Clay", "Sand", "Silt", "Coarse"])
    raw = raw.merge(tex, left_on="POINTID", right_on="Point_ID", how="left")

    year = pd.to_datetime(raw["SURVEY_DATE"], format="%d-%m-%y", errors="coerce").dt.year
    return pd.DataFrame(
        {
            "source": "lucas_2018",
            "sample_id": raw["POINTID"].astype(str),
            "latitude": raw["TH_LAT"],
            "longitude": raw["TH_LONG"],
            "country": raw["NUTS_0"],
            "continent": "Europe",
            "sample_year": year,
            "ph_h2o": raw["pH_H2O"],
            "ph_cacl2": raw["pH_CaCl2"],
            "oc_gkg": raw["OC"],
            "oc_20_30_gkg": raw["OC (20-30 cm)"],
            "n_gkg": raw["N"],
            "p_mgkg": raw["P"],
            "k_mgkg": raw["K"],
            "ec": raw["EC"],
            "caco3": raw["CaCO3"],
            "caco3_20_30": raw["CaCO3 (20-30 cm)"],
            "ox_al": raw["Ox_Al"],
            "ox_fe": raw["Ox_Fe"],
            "clay_pct": raw["Clay"],
            "sand_pct": raw["Sand"],
            "silt_pct": raw["Silt"],
            "coarse_pct": raw["Coarse"],
            "elevation_m": raw["Elev"],
            "bd_0_20": raw["bd_0_20"] if "bd_0_20" in raw.columns else np.nan,
            "land_cover": raw["LC0_Desc"],
            "land_cover_l1": raw["LC1_Desc"],
            "land_use": raw["LU1_Desc"],
            "depth": raw["Depth"],
            "nuts1": raw["NUTS_1"],
            "nuts2": raw["NUTS_2"],
            "nuts3": raw["NUTS_3"],
        }
    )


def load_lucas_2009() -> pd.DataFrame:
    """1st LUCAS soil campaign. Has its own coordinates (unlike 2015)."""
    raw = pd.read_excel(LUCAS_2009)
    _numeric_cols(raw, ["GPS_LAT", "GPS_LONG", "pH_in_H2O", "pH_in_CaCl2",
                         "OC", "CaCO3", "N", "P", "K", "CEC", "clay", "sand", "silt", "coarse"])
    return pd.DataFrame(
        {
            "source": "lucas_2009",
            "sample_id": raw["POINT_ID"].astype(str),
            "latitude": raw["GPS_LAT"],
            "longitude": raw["GPS_LONG"],
            "country": pd.NA,
            "continent": "Europe",
            "sample_year": 2009,
            "ph_h2o": raw["pH_in_H2O"],
            "ph_cacl2": raw["pH_in_CaCl2"],
            "oc_gkg": raw["OC"],
            "n_gkg": raw["N"],
            "p_mgkg": raw["P"],
            "k_mgkg": raw["K"],
            "caco3": raw["CaCO3"],
            "clay_pct": raw["clay"],
            "sand_pct": raw["sand"],
            "silt_pct": raw["silt"],
            "coarse_pct": raw["coarse"],
            "cec_ph7": raw["CEC"],
        }
    )


def load_lucas_2015() -> pd.DataFrame:
    """Extra rows: 2015 campaign, including points that were not resurveyed in 2018."""
    raw = pd.read_csv(LUCAS_2015, low_memory=False)
    _numeric_cols(
        raw,
        ["Clay", "Sand", "Silt", "Coarse", "pH(CaCl2)", "pH(H2O)", "EC", "OC", "CaCO3", "P", "N", "K", "Elevation"],
    )
    return pd.DataFrame(
        {
            "source": "lucas_2015",
            "sample_id": raw["Point_ID"].astype(str),
            "latitude": np.nan,
            "longitude": np.nan,
            "country": raw["NUTS_0"],
            "continent": "Europe",
            "sample_year": 2015,
            "ph_h2o": raw["pH(H2O)"],
            "ph_cacl2": raw["pH(CaCl2)"],
            "oc_gkg": raw["OC"],
            "n_gkg": raw["N"],
            "p_mgkg": raw["P"],
            "k_mgkg": raw["K"],
            "ec": raw["EC"],
            "caco3": raw["CaCO3"],
            "clay_pct": raw["Clay"],
            "sand_pct": raw["Sand"],
            "silt_pct": raw["Silt"],
            "coarse_pct": raw["Coarse"],
            "elevation_m": raw["Elevation"],
            "land_cover": raw["LC1_Desc"],
            "land_use": raw["LU1_Desc"],
            "soil_stones": raw["Soil_Stones"],
            "nuts1": raw["NUTS_1"],
            "nuts2": raw["NUTS_2"],
            "nuts3": raw["NUTS_3"],
        }
    )


def load_wosis_property(path: Path, stem: str, max_upper_depth: int = 30) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", usecols=WOSIS_COLS, chunksize=150_000, low_memory=False):
        top = chunk[(chunk["upper_depth"] < max_upper_depth) & chunk["value_avg"].notna()]
        chunks.append(top)
    if not chunks:
        return pd.DataFrame(columns=["profile_id", "longitude", "latitude", "country_name", "continent", "date", stem])
    raw = pd.concat(chunks, ignore_index=True)
    raw = raw.sort_values(["profile_id", "upper_depth"]).drop_duplicates("profile_id", keep="first")
    raw = raw.rename(columns={"value_avg": stem})
    return raw[["profile_id", "longitude", "latitude", "country_name", "continent", "date", stem]]


def wosis_property_files() -> list[tuple[str, Path]]:
    files = []
    for path in sorted(WOSIS_DIR.glob("wosis_202312_*.tsv")):
        stem = path.stem.replace("wosis_202312_", "")
        if stem not in WOSIS_SKIP:
            files.append((stem, path))
    return files


def load_wosis() -> pd.DataFrame:
    files = wosis_property_files()
    if not files:
        return pd.DataFrame()

    base = None
    meta = ["longitude", "latitude", "country_name", "continent", "date"]
    for stem, path in files:
        table = load_wosis_property(path, stem)
        if base is None:
            base = table
            continue
        base = base.merge(table, on="profile_id", how="outer", suffixes=("", "_new"))
        for col in meta:
            new_col = f"{col}_new"
            if new_col in base.columns:
                base[col] = base[col].combine_first(base[new_col])
                base = base.drop(columns=[new_col])
        extra = f"{stem}_new"
        if extra in base.columns:
            base[stem] = base[stem].combine_first(base[extra]) if stem in base.columns else base[extra]
            base = base.drop(columns=[extra])

    year = pd.to_datetime(base["date"], errors="coerce").dt.year
    out = pd.DataFrame(
        {
            "source": "wosis",
            "sample_id": "wosis_" + base["profile_id"].astype("Int64").astype(str),
            "latitude": base["latitude"],
            "longitude": base["longitude"],
            "country": base["country_name"],
            "continent": base["continent"],
            "sample_year": year,
        }
    )
    for stem, _ in files:
        col = WOSIS_CANONICAL.get(stem, f"wosis_{stem}")
        if col in out.columns:
            out[col] = out[col].combine_first(base[stem])
        else:
            out[col] = base[stem]
    return out


def _fill_lucas_2015_coords(lucas15: pd.DataFrame, lucas18: pd.DataFrame) -> pd.DataFrame:
    """2015 file has no lat/lon; copy coordinates from the same LUCAS point in 2018."""
    coords = lucas18[["sample_id", "latitude", "longitude"]].drop_duplicates("sample_id")
    merged = lucas15.drop(columns=["latitude", "longitude"]).merge(coords, on="sample_id", how="left")
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    if "ph_h2o" in df.columns:
        df.loc[~df["ph_h2o"].between(3.0, 11.0), "ph_h2o"] = np.nan
    if "oc_gkg" in df.columns:
        df.loc[(df["oc_gkg"] < 0) | (df["oc_gkg"] > 400), "oc_gkg"] = np.nan
    if "n_gkg" in df.columns:
        df.loc[(df["n_gkg"] < 0) | (df["n_gkg"] > 50), "n_gkg"] = np.nan
    for tex in ["clay_pct", "sand_pct", "silt_pct", "coarse_pct"]:
        if tex in df.columns:
            df.loc[(df[tex] < 0) | (df[tex] > 100), tex] = np.nan

    if {"clay_pct", "sand_pct", "silt_pct"}.issubset(df.columns):
        tex_sum = df["clay_pct"] + df["sand_pct"] + df["silt_pct"]
        bad_tex = tex_sum.notna() & ((tex_sum < 85) | (tex_sum > 115))
        df.loc[bad_tex, ["clay_pct", "sand_pct", "silt_pct"]] = np.nan

    chemistry = [c for c in ["ph_h2o", "oc_gkg", "n_gkg", "clay_pct"] if c in df.columns]
    df = df.dropna(subset=chemistry, how="all")
    df = df.drop_duplicates(subset=["source", "sample_id"], keep="first")

    if "ph_h2o" in df.columns:
        df["acidic"] = np.where(df["ph_h2o"].notna(), (df["ph_h2o"] < 5.5).astype(int), np.nan)
    if "oc_gkg" in df.columns:
        df["low_oc"] = np.where(df["oc_gkg"].notna(), (df["oc_gkg"] < 10).astype(int), np.nan)
    if "acidic" in df.columns and "low_oc" in df.columns:
        df["soil_stress"] = np.where(
            df["acidic"].isna() & df["low_oc"].isna(),
            np.nan,
            ((df["acidic"].fillna(0) == 1) | (df["low_oc"].fillna(0) == 1)).astype(float),
        )
    return df.reset_index(drop=True)


def build_training_csv(out_path: Path | None = None) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    lucas18 = load_lucas_2018()
    lucas15 = _fill_lucas_2015_coords(load_lucas_2015(), lucas18)
    lucas09 = load_lucas_2009()
    wosis = load_wosis()
    combined = pd.concat([lucas18, lucas15, lucas09, wosis], ignore_index=True, sort=False)
    final = clean(combined)
    path = out_path or TRAINING_CSV
    final.to_csv(path, index=False)
    return final


def load_training_table(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not TRAINING_CSV.exists():
        return build_training_csv()
    return pd.read_csv(TRAINING_CSV, low_memory=False)