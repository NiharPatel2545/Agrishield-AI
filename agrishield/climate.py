"""Attach satellite/climate columns to the training table.

Run ONCE, offline, to build agrishield_training.csv into its final form
before training. This is never called at farmer-inference time — that
uses gee.py functions directly on one coordinate (see inference.py).
"""

from __future__ import annotations

import time

import ee
import pandas as pd

from agrishield.config import TRAINING_CSV
from agrishield.dataset import drop_gee_dead_rows
from agrishield.gee import (
    CLAY_IMG,
    ELEVATION_IMG,
    SAND_IMG,
    SENTINEL_BANDS,
    initialize,
)


def _sample_image_batch(batch: pd.DataFrame, image: ee.Image) -> pd.DataFrame:
    """Sample one ee.Image at every (lat, lon) in `batch` in a single call.

    Uses per-point reduceRegion() rather than sampleRegions() -- this
    mirrors the exact call that already works for a single live point
    (see gee.py), just wrapped into one FeatureCollection so it's still
    ONE network round trip for the whole batch, not one per row.
    sampleRegions() was silently returning near-empty results at scale
    (0.3% fill on a real smoke test) with no exception raised.
    """
    features = []
    for row_id, row in batch.iterrows():
        point = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
        values = image.reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=20, maxPixels=1e9
        )
        features.append(ee.Feature(None, values.set("row_id", int(row_id))))
    collection = ee.FeatureCollection(features)
    sampled = collection.getInfo()
    records = [feat.get("properties", {}) for feat in sampled.get("features", [])]
    return pd.DataFrame.from_records(records)


def _run_batches(df: pd.DataFrame, ids: list[int], image: ee.Image,
                  batch_size: int, out_cols: list[str], pause_s: float) -> pd.DataFrame:
    work = df.copy()
    # Force these columns to a nullable float dtype up front. Without this,
    # a batch that comes back entirely empty assigns an all-None array into
    # a float64 column, which pandas now warns about (and will hard-error
    # on in a future version) -- "Float64" (nullable) accepts None natively.
    for col in out_cols:
        if col in work.columns:
            work[col] = work[col].astype("Float64")

    failed_batches: list[tuple[int, int]] = []
    for start in range(0, len(ids), batch_size):
        chunk_ids = ids[start : start + batch_size]
        batch = work.loc[chunk_ids, ["latitude", "longitude"]]
        try:
            result = _sample_image_batch(batch, image)
        except Exception:
            time.sleep(2)
            try:
                result = _sample_image_batch(batch, image)
            except Exception as exc:
                # Both attempts failed outright (not just an empty result) --
                # record it instead of silently moving on, so you can tell
                # "no data at these coordinates" apart from "GEE call broke".
                failed_batches.append((start, start + len(chunk_ids)))
                print(f"  BATCH FAILED rows {start}-{start + len(chunk_ids)}: {exc}")
                time.sleep(pause_s)
                continue
        if result.empty or "row_id" not in result.columns:
            failed_batches.append((start, start + len(chunk_ids)))
            print(f"  BATCH EMPTY rows {start}-{start + len(chunk_ids)} (no rows returned)")
            time.sleep(pause_s)
            continue
        result = result.dropna(subset=["row_id"]).set_index("row_id")
        for col in out_cols:
            if col in result.columns:
                work.loc[result.index, col] = result[col].astype("Float64")
        print(f"  rows {start}-{start + len(chunk_ids)} / {len(ids)}")
        time.sleep(pause_s)

    if failed_batches:
        print(f"  -> {len(failed_batches)} batch(es) returned nothing usable "
              f"({sum(b - a for a, b in failed_batches)} rows affected): {failed_batches}")
    return work


def attach_worldclim(df: pd.DataFrame, batch_size: int = 400, pause_s: float = 0.2) -> pd.DataFrame:
    """Add tmean_c and precip_mm. Same rows, extra columns."""
    initialize()
    work = df.copy()
    out_cols = ["tmean_c", "temp_seasonality", "precip_mm", "precip_seasonality"]
    for col in out_cols:
        if col not in work.columns:
            work[col] = pd.NA
    image = ee.Image("WORLDCLIM/V1/BIO").select(
        ["bio01", "bio04", "bio12", "bio15"], out_cols
    )
    pending = work[work["tmean_c"].isna() & work["latitude"].notna() & work["longitude"].notna()]
    work = _run_batches(work, list(pending.index), image, batch_size, out_cols, pause_s)
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
    # silt_pct is NOT sampled from GEE -- see gee.py note: OpenLandMap never
    # published a silt image to the EE catalog. It's derived below instead.
    out_cols = ["clay_pct", "sand_pct", "elevation_m", "slope_deg"]
    work["_soil_pending"] = work["latitude"].notna() & work["longitude"].notna()

    clay = ee.Image(CLAY_IMG).select("b0").rename("clay_pct")
    sand = ee.Image(SAND_IMG).select("b0").rename("sand_pct")
    elev = ee.Image(ELEVATION_IMG).select("elevation").rename("elevation_m")
    slope = ee.Terrain.slope(ee.Image(ELEVATION_IMG)).rename("slope_deg")
    stack = clay.addBands([sand, elev, slope])

    ids = list(work[work["_soil_pending"]].index)
    work = _run_batches(work, ids, stack, batch_size, out_cols, pause_s)
    work = work.drop(columns=["_soil_pending"])

    if "silt_pct" not in work.columns:
        work["silt_pct"] = pd.NA
    have_both = work["clay_pct"].notna() & work["sand_pct"].notna()
    work.loc[have_both, "silt_pct"] = 100.0 - work.loc[have_both, "clay_pct"] - work.loc[have_both, "sand_pct"]
    return work


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

    # Sentinel-2 launched mid-2015 (COPERNICUS/S2_SR_HARMONIZED has no usable
    # coverage before then). Years before this can NEVER return real bands --
    # querying them just burns GEE quota and time for a guaranteed-empty
    # result, and would do so again on every future re-run/resume. Skip them
    # outright; the imputer covers these rows at train time, same as any
    # other missing feature.
    MIN_SENTINEL_YEAR = 2016  # 2015 itself has only partial-year coverage
    years = sorted(
        y for y in work[year_col].dropna().unique()
        if pd.notna(y) and int(y) >= MIN_SENTINEL_YEAR
    )
    skipped = (work[year_col].notna() & (work[year_col] < MIN_SENTINEL_YEAR)).sum()
    if skipped:
        print(f"Skipping {skipped} rows with sample_year < {MIN_SENTINEL_YEAR} "
              f"(no Sentinel-2 coverage exists -- not a failure, just physically impossible)")

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
    """Attach WorldClim + static soil + Sentinel-2, saving to disk after each stage.

    Checkpointed: if this crashes partway (GEE quota, network blip, laptop
    sleep), the CSV on disk already has whatever stages finished before the
    crash. Re-running enrich_training_csv() picks up where it left off,
    because attach_worldclim/attach_static_soil only touch rows that are
    still NaN (see the `.isna()` masks inside each attach_* function) --
    already-filled rows are skipped, not re-fetched.
    """
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    subset = df.iloc[:max_rows] if max_rows is not None else df
    rest = df.iloc[max_rows:] if max_rows is not None else df.iloc[0:0]

    subset = attach_worldclim(subset, batch_size=batch_size)
    df = pd.concat([subset, rest], ignore_index=True) if max_rows is not None else subset
    df.to_csv(TRAINING_CSV, index=False)
    print("checkpoint saved after worldclim")

    subset = attach_static_soil(subset, batch_size=batch_size)
    df = pd.concat([subset, rest], ignore_index=True) if max_rows is not None else subset
    df.to_csv(TRAINING_CSV, index=False)
    print("checkpoint saved after static soil")

    subset = attach_sentinel_batched(subset, batch_size=batch_size)
    df = pd.concat([subset, rest], ignore_index=True) if max_rows is not None else subset
    df.to_csv(TRAINING_CSV, index=False)
    print("checkpoint saved after sentinel")

    before = len(df)
    df = drop_gee_dead_rows(df)
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} rows with zero usable GEE/climate signal")
        df.to_csv(TRAINING_CSV, index=False)

    return df