from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from agrishield.config import FEATURE_COLUMNS, MODEL_PATH, MODELS_DIR, TARGET_COLUMN

# Found by RandomizedSearchCV (30 candidates x 5-fold GroupKFold, scoring="f1"
# on the acidic class, grouped by sample_id) -- see git history / project
# notes for the search script if these ever need retuning. Result: CV f1 =
# 0.721, and importantly max_depth landed low (7, out of a 3-8 search range)
# -- shallow trees are what actually helped continent-holdout generalization,
# not just in-distribution accuracy. Don't casually raise max_depth back up
# without rerunning the continent holdout check below.
_TUNED_XGB_PARAMS = dict(
    n_estimators=407,
    max_depth=7,
    learning_rate=0.1966,
    subsample=0.6241,
    colsample_bytree=0.8448,
    min_child_weight=9,
)


def _pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                xgb.XGBClassifier(
                    **_TUNED_XGB_PARAMS,
                    scale_pos_weight=scale_pos_weight,  # XGBoost has no
                    # class_weight="balanced" -- this is the equivalent.
                    # Computed per-call from the actual training split
                    # (neg/pos), not hardcoded, so retraining on updated
                    # data keeps this correctly calibrated to whatever the
                    # class balance is at the time, same principle as
                    # class_weight="balanced" for RandomForest.
                    random_state=0,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def train(df: pd.DataFrame, target: str = TARGET_COLUMN, group_col: str = "sample_id") -> tuple[Pipeline, str]:
    """Train on chemistry/texture/climate. pH is the label source, so it is not a feature.

    Split by `group_col` (sample_id), not a plain random split. LUCAS
    resurveys the same physical points across 2009/2015/2018 -- a random
    split lets the same location land in both train and test, which
    inflates the reported accuracy (the model has effectively already
    "seen" that test point). GroupShuffleSplit keeps every row for a given
    sample_id entirely on one side of the split.
    """
    cols = available_features(df)
    work = df.dropna(subset=[target]).copy()
    if group_col not in work.columns:
        raise ValueError(f"'{group_col}' column required for a leakage-safe split")

    print("class balance (target = %s):" % target)
    print(work[target].value_counts(normalize=True).round(3).to_string())

    X = work[cols]
    y = work[target]
    groups = work[group_col]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = _pipeline(scale_pos_weight=neg / pos)
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), digits=3)
    return model, report


def continent_holdout_check(df: pd.DataFrame, target: str = TARGET_COLUMN,
                             min_test_rows: int = 100) -> None:
    """Diagnostic, not a fix: train with one continent fully removed, test only on it.

    Loops every continent with enough rows, not just one pair -- a single
    biggest-vs-second-biggest comparison (as train_pipeline.py used to do)
    hides how badly generalization varies by region. This is exactly the
    check that showed cross-continent recall dropping to 0.25-0.40 versus
    0.665 in-distribution on the RandomForest model; rerun this after any
    retrain to confirm XGBoost's continent numbers before trusting them.
    """
    if "continent" not in df.columns:
        print("no 'continent' column -- skipping holdout check")
        return
    cols = available_features(df)
    data = df.dropna(subset=[target])
    for holdout in data["continent"].dropna().unique():
        tr = data[data["continent"] != holdout]
        te = data[data["continent"] == holdout]
        if len(te) < min_test_rows:
            print(f"skipping {holdout!r}: only {len(te)} rows (< {min_test_rows})")
            continue
        neg, pos = (tr[target] == 0).sum(), (tr[target] == 1).sum()
        pipe = _pipeline(scale_pos_weight=neg / pos)
        pipe.fit(tr[cols], tr[target])
        preds = pipe.predict(te[cols])
        print(f"\n--- trained WITHOUT {holdout}, tested ON {holdout} (n={len(te)}) ---")
        print(classification_report(te[target], preds, digits=3))


def save_model(model: Pipeline, path: Path | None = None) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    out = path or MODEL_PATH
    joblib.dump(model, out)
    return out


def load_model(path: Path | None = None) -> Pipeline:
    return joblib.load(path or MODEL_PATH)


def predict_proba(model: Pipeline, features: dict) -> dict:
    cols = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    frame = pd.DataFrame([{col: features.get(col) for col in cols}])
    proba = model.predict_proba(frame)[0]
    classes = list(model.named_steps["clf"].classes_)
    predicted = int(model.predict(frame)[0])
    return {
        "predicted": predicted,
        "probability": {int(c): float(p) for c, p in zip(classes, proba)},
    }