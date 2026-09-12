from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from agrishield.config import FEATURE_COLUMNS, MODEL_PATH, MODELS_DIR, TARGET_COLUMN


def _pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
        ]
    )


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def train(df: pd.DataFrame, target: str = TARGET_COLUMN) -> tuple[Pipeline, str]:
    """Train on chemistry/texture/climate. pH is the label source, so it is not a feature."""
    cols = available_features(df)
    work = df.dropna(subset=[target]).copy()
    X = work[cols]
    y = work[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
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
