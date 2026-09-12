"""Attach WorldClim climate as extra columns on existing training rows."""

from __future__ import annotations

import time

import ee
import pandas as pd

from agrishield.config import TRAINING_CSV
from agrishield.gee import initialize


def _sample_batch(batch: pd.DataFrame) -> pd.DataFrame:
    features = []
    for row_id, row in batch.iterrows():
        geom = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
        features.append(ee.Feature(geom, {"row_id": int(row_id)}))
    collection = ee.FeatureCollection(features)
    image = ee.Image("WORLDCLIM/V1/BIO").select(["bio01", "bio12"])
    sampled = image.sampleRegions(collection=collection, scale=1000, geometries=False).getInfo()
    records = []
    for feat in sampled.get("features", []):
        props = feat.get("properties", {})
        tmean = props.get("bio01")
        records.append(
            {
                "row_id": props.get("row_id"),
                "tmean_c": None if tmean is None else tmean / 10.0,
                "precip_mm": props.get("bio12"),
            }
        )
    return pd.DataFrame.from_records(records)


def attach_worldclim(df: pd.DataFrame, batch_size: int = 400, pause_s: float = 0.2) -> pd.DataFrame:
    """Add tmean_c and precip_mm. Same rows, extra columns. Does not add labelled soil rows."""
    initialize()
    work = df.copy()
    if "tmean_c" not in work.columns:
        work["tmean_c"] = pd.NA
        work["precip_mm"] = pd.NA

    pending = work[work["tmean_c"].isna() & work["latitude"].notna() & work["longitude"].notna()]
    ids = list(pending.index)
    for start in range(0, len(ids), batch_size):
        chunk_ids = ids[start : start + batch_size]
        batch = work.loc[chunk_ids, ["latitude", "longitude"]]
        try:
            clim = _sample_batch(batch)
        except Exception:
            time.sleep(2)
            clim = _sample_batch(batch)
        if clim.empty:
            continue
        clim = clim.dropna(subset=["row_id"]).set_index("row_id")
        work.loc[clim.index, "tmean_c"] = clim["tmean_c"]
        work.loc[clim.index, "precip_mm"] = clim["precip_mm"]
        time.sleep(pause_s)
    return work


def enrich_training_csv(batch_size: int = 400, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    if max_rows is not None:
        subset = df.iloc[:max_rows]
        rest = df.iloc[max_rows:]
        subset = attach_worldclim(subset, batch_size=batch_size)
        df = pd.concat([subset, rest], ignore_index=True)
    else:
        df = attach_worldclim(df, batch_size=batch_size)
    df.to_csv(TRAINING_CSV, index=False)
    return df
