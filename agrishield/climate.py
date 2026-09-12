from __future__ import annotations

import time

import ee
import pandas as pd

from agrishield.config import TRAINING_CSV
from agrishield.gee import (
    CLAY_IMG,
    ELEVATION_IMG,
    SAND_IMG,
    SENTINEL_BANDS,
    SILT_IMG,
    initialize,
)


def _sample_image_batch(batch: pd.DataFrame, image: ee.Image) -> pd.DataFrame:
    """Sample one ee.Image at every (lat, lon) in `batch` in a single call."""
    features = []
    for row_id, row in batch.iterrows():
        geom = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
        features.append(ee.Feature(geom, {"row_id": int(row_id)}))
    collection = ee.FeatureCollection(features)
    sampled = image.sampleRegions(collection=collection, scale=250, geometries=False).getInfo()
    records = [feat.get("properties", {}) for feat in sampled.get("features", [])]
    return pd.DataFrame.from_records(records)


def _run_batches(df: pd.DataFrame, ids: list[int], image: ee.Image,
                  batch_size: int, out_cols: list[str], pause_s: float) -> pd.DataFrame:
    work = df.copy()
    for start in range(0, len(ids), batch_size):
        chunk_ids = ids[start : start + batch_size]
        batch = work.loc[chunk_ids, ["latitude", "longitude"]]
        try:
            result = _sample_image_batch(batch, image)
        except Exception:
            time.sleep(2)
            result = _sample_image_batch(batch, image)
        if result.empty or "row_id" not in result.columns:
            continue
        result = result.dropna(subset=["row_id"]).set_index("row_id")
        for col in out_cols:
            if col in result.columns:
                work.loc[result.index, col] = result[col]
        print(f"  rows {start}-{start + len(chunk_ids)} / {len(ids)}", end="\r")
        time.sleep(pause_s)
    return work


def attach_worldclim(df: pd.DataFrame, batch_size: int = 400, pause_s: float = 0.2) -> pd.DataFrame:
    """Add tmean_c and precip_mm. Same rows, extra columns."""
    initialize()
    work = df.copy()
    if "tmean_c" not in work.columns:
        work["tmean_c"] = pd.NA
        work["precip_mm"] = pd.NA
    image = ee.Image("WORLDCLIM/V1/BIO").select(["bio01", "bio12"], ["tmean_c", "precip_mm"])
    pending = work[work["tmean_c"].isna() & work["latitude"].notna() & work["longitude"].notna()]
    work = _run_batches(work, list(pending.index), image, batch_size, ["tmean_c", "precip_mm"], pause_s)
    # bio01 is temp * 10 in the raw layer; the renamed sample already applies no scaling,
    # so divide here once, after sampling.
    mask = work["tmean_c"].notna()
    work.loc[mask, "tmean_c"] = work.loc[mask, "tmean_c"] / 10.0
    return work


def attach_static_soil(df: pd.DataFrame, batch_size: int = 400, pause_s: float = 0.2) -> pd.DataFrame:
    """OVERWRITE clay_pct, sand_pct, silt_pct, elevation_m with GEE values.

    Deliberately overwrites the LUCAS lab/survey columns of the same name,
    rather than sitting beside them under a different name. Inference will
    always source these from satellite (see gee.static_soil_at_point), so
    training must use the exact same source -- otherwise you're right back
    to the train/serve mismatch, just hidden under a new column name.
    Original lab values are still visible in your source CSVs if you ever
    want to compare accuracy against ground truth.
    """
    initialize()
    work = df.copy()
    out_cols = ["clay_pct", "sand_pct", "silt_pct", "elevation_m"]
    work["_soil_pending"] = work["latitude"].notna() & work["longitude"].notna()

    clay = ee.Image(CLAY_IMG).select("b0").rename("clay_pct")
    sand = ee.Image(SAND_IMG).select("b0").rename("sand_pct")
    silt = ee.Image(SILT_IMG).select("b0").rename("silt_pct")
    elev = ee.Image(ELEVATION_IMG).select("elevation").rename("elevation_m")
    stack = clay.addBands([sand, silt, elev])

    ids = list(work[work["_soil_pending"]].index)
    work = _run_batches(work, ids, stack, batch_size, out_cols, pause_s)
    return work.drop(columns=["_soil_pending"])


def attach_sentinel_batched(
    df: pd.DataFrame,
    batch_size: int = 400,
    pause_s: float = 0.3,
    year_col: str = "sample_year",
) -> pd.DataFrame:
    """Add B2/B3/B4/B8/B11/ndvi, grouped by sample_year.

    Instead of one live GEE call per row (230k rows x ~1-2s = days), this
    builds ONE cloud-free median composite per calendar year, then batches
    all rows from that year through sampleRegions(). Total calls drop from
    ~230,000 to roughly (num_years x rows_per_year / batch_size) -- usually
    well under 500 calls total, finishing in well under an hour.

    Trade-off: loses exact +/-15-day alignment to each row's own survey
    date in favour of a full-year composite. For background spectral
    signature this is a reasonable trade -- a full-year median is also
    less sensitive to one unlucky cloudy week than a narrow window is.
    """
    initialize()
    work = df.copy()
    out_cols = [*SENTINEL_BANDS, "ndvi"]
    for col in out_cols:
        if col not in work.columns:
            work[col] = pd.NA

    years = sorted(
        y for y in work[year_col].dropna().unique()
        if pd.notna(y)
    )
    for year in years:
        year = int(year)
        start, end = f"{year}-01-01", f"{year}-12-31"
        composite = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .select(SENTINEL_BANDS)
            .median()
        )
        ndvi = composite.normalizedDifference(["B8", "B4"]).rename("ndvi")
        image = composite.addBands(ndvi)

        year_mask = (
            (work[year_col] == year)
            & work[out_cols[0]].isna()
            & work["latitude"].notna()
            & work["longitude"].notna()
        )
        ids = list(work[year_mask].index)
        if not ids:
            continue
        print(f"Year {year}: {len(ids)} rows")
        work = _run_batches(work, ids, image, batch_size, out_cols, pause_s)
    return work


def enrich_training_csv(batch_size: int = 400, max_rows: int | None = None) -> pd.DataFrame:
    """Attach WorldClim + static soil + Sentinel-2, then overwrite the CSV."""
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    subset = df.iloc[:max_rows] if max_rows is not None else df
    rest = df.iloc[max_rows:] if max_rows is not None else df.iloc[0:0]

    subset = attach_worldclim(subset, batch_size=batch_size)
    subset = attach_static_soil(subset, batch_size=batch_size)
    subset = attach_sentinel_batched(subset, batch_size=batch_size)

    df = pd.concat([subset, rest], ignore_index=True) if max_rows is not None else subset
    df.to_csv(TRAINING_CSV, index=False)
    return df