"""Main entry point: build -> enrich -> train -> save.

Run stages independently the first time (GEE enrichment is slow and costs
quota), then use --skip-enrich on reruns once data/agrishield_training.csv
already has satellite/climate columns filled in.

    python train_pipeline.py                 # build + enrich + train
    python train_pipeline.py --skip-enrich    # train only, reuse existing CSV
    python train_pipeline.py --skip-build --skip-enrich  # train only, as-is
"""

from __future__ import annotations

import argparse

import pandas as pd

from agrishield.config import TRAINING_CSV, FEATURE_COLUMNS
from agrishield.dataset import build_training_csv, drop_gee_dead_rows
from agrishield.climate import enrich_training_csv
from agrishield.model import train, save_model, available_features


def continent_holdout_check(df: pd.DataFrame, target: str = "acidic") -> None:
    """Diagnostic, not a fix: trains on one continent, tests on another.

    climate/texture columns can act as a location fingerprint even with
    lat/lon dropped (see problem #2) -- if accuracy collapses when tested
    on a continent the model never trained on, that's a strong sign the
    model is leaning on regional signature rather than universal soil
    physics. This does not change training; it just tells you whether to
    worry.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline

    if "continent" not in df.columns:
        print("no 'continent' column -- skipping holdout check")
        return

    work = df.dropna(subset=[target]).copy()
    counts = work["continent"].value_counts()
    if len(counts) < 2:
        print("only one continent present -- skipping holdout check")
        return

    biggest, second = counts.index[0], counts.index[1]
    cols = available_features(work)

    train_df = work[work["continent"] == biggest]
    test_df = work[work["continent"] == second]
    if train_df.empty or test_df.empty:
        return

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, class_weight="balanced")),
        ]
    )
    model.fit(train_df[cols], train_df[target])
    preds = model.predict(test_df[cols])
    print(f"\n--- continent holdout: train={biggest} ({len(train_df)} rows) "
          f"-> test={second} ({len(test_df)} rows) ---")
    print(classification_report(test_df[target], preds, digits=3))
    print("(compare this report's numbers to the main grouped-split report above;"
          " a big drop here means the model is leaning on regional signature)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true", help="reuse existing raw merge, skip LUCAS/WoSIS reload")
    parser.add_argument("--skip-enrich", action="store_true", help="skip GEE calls, assume CSV already has satellite/climate cols")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--max-rows", type=int, default=None, help="cap rows during enrichment for a smoke test")
    parser.add_argument("--skip-holdout-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        print("=== building raw LUCAS + WoSIS merge ===")
        df = build_training_csv()
        print("rows:", len(df))
        print(df["source"].value_counts().to_string())
    else:
        df = pd.read_csv(TRAINING_CSV, low_memory=False)

    if not args.skip_enrich:
        print("\n=== enriching with GEE (Sentinel-2 / WorldClim / static soil) ===")
        df.to_csv(TRAINING_CSV, index=False)  # enrich_training_csv reads from disk
        df = enrich_training_csv(batch_size=args.batch_size, max_rows=args.max_rows)
    else:
        df = pd.read_csv(TRAINING_CSV, low_memory=False)
        before = len(df)
        df = drop_gee_dead_rows(df)
        if len(df) != before:
            print(f"dropped {before - len(df)} rows with zero usable GEE signal")

    feature_cover = df[available_features(df)].notna().mean().round(3)
    print("\nfeature coverage (fraction non-null):")
    print(feature_cover.to_string())

    sentinel_cols = [c for c in ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12", "ndvi"] if c in FEATURE_COLUMNS]
    missing_sentinel = [c for c in sentinel_cols if c not in df.columns]
    if missing_sentinel:
        raise RuntimeError(
            f"Sentinel columns missing entirely from the training table: {missing_sentinel}. "
            "This means enrich_training_csv() never finished/checkpointed the sentinel stage "
            "for this CSV -- training would silently proceed with zero satellite signal. "
            "Re-run without --skip-enrich, or --skip-build --skip-enrich only once you've "
            "confirmed `checkpoint saved after sentinel` printed and B2 etc. are in the CSV."
        )
    zero_cover = [c for c in sentinel_cols if feature_cover.get(c, 0) == 0]
    if zero_cover:
        print(f"\nWARNING: these Sentinel columns exist but are 100% null: {zero_cover} -- "
              "the model will train on them but they contribute nothing but imputed medians.")

    print("\n=== training (grouped split by sample_id) ===")
    model, report = train(df)
    print(report)

    if not args.skip_holdout_check:
        continent_holdout_check(df)

    path = save_model(model)
    print(f"\nsaved model to {path}")


if __name__ == "__main__":
    main()