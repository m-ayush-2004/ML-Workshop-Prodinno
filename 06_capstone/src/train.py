"""
Train an XGBoost diabetes-risk classifier and save it as a new, timestamped
model version.

Usage (from anywhere, paths are resolved relative to this file):
    python src/train.py
    python src/train.py --data data/diabetes.csv --note "baseline"

Every run creates a brand-new `models/v_<YYYYMMDD_HHMMSS>/` folder — nothing is
ever overwritten. `models/registry.json` is appended to and its "latest"
pointer is updated so the API/UI always know which version is current, while
every previous version remains on disk and in the registry for comparison or
rollback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data_processing import FEATURE_COLUMNS, clean, get_feature_target, load_raw  # noqa: E402

DEFAULT_DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


def _timestamp_version(now: str) -> str:
    return f"v_{now}"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"versions": [], "latest": None}


def _save_registry(registry: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def train_and_save(now: str, data_path: Path = DEFAULT_DATA_PATH, note: str = "", extra_rows_path: str | None = None) -> dict:
    """Train one new model version and register it. `now` is an externally
    supplied YYYYMMDD_HHMMSS timestamp so the caller controls how "current
    time" is produced (see the note in api/main.py about avoiding datetime.now
    inside library code that might run under a workflow)."""
    df = load_raw(data_path)
    if extra_rows_path:
        extra = load_raw(extra_rows_path)
        df = pd.concat([df, extra], ignore_index=True)

    df_clean, impute_stats = clean(df)
    X, y = get_feature_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_total_rows": int(len(df)),
    }

    version = _timestamp_version(now)
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.joblib")
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (version_dir / "impute_stats.json").write_text(json.dumps(impute_stats, indent=2))
    (version_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    # Small background sample for LIME's local surrogate model at explain-time.
    X_train.sample(n=min(100, len(X_train)), random_state=42).to_csv(
        version_dir / "background_sample.csv", index=False
    )

    registry = _load_registry()
    registry["versions"].append(
        {
            "version": version,
            "timestamp": now,
            "note": note,
            "metrics": metrics,
            "path": version_dir.relative_to(BASE_DIR).as_posix(),
        }
    )
    registry["latest"] = version
    _save_registry(registry)

    return {"version": version, "metrics": metrics}


def ensure_initial_model(now: str) -> dict:
    """Bootstrap: if no version exists yet (fresh clone / fresh container),
    train the first one automatically so the API/UI work immediately."""
    registry = _load_registry()
    if registry["versions"]:
        return {"version": registry["latest"], "metrics": _load_registry()["versions"][-1]["metrics"], "bootstrapped": False}
    result = train_and_save(now=now, note="initial bootstrap model")
    result["bootstrapped"] = True
    return result


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Train a new diabetes-risk model version.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--note", type=str, default="manual train.py run")
    args = parser.parse_args()

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = train_and_save(now=now, data_path=Path(args.data), note=args.note)
    print(json.dumps(result, indent=2))
