from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from agrishield.config import FEATURE_COLUMNS, MODEL_PATH, MODELS_DIR, TARGET_COLUMN


def _pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=0,
                    n_jobs=-1,
                    class_weight="balanced",  # was unweighted -- if `acidic`
                    # isn't ~50/50, an unweighted RF can look accurate while
                    # just learning the majority class. class_weight fixes
                    # the loss; you should still eyeball the printed balance
                    # below rather than trust accuracy alone.
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

    model = _pipeline()
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), digits=3)
    return model, report


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