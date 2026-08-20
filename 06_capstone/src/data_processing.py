"""
Shared data loading / cleaning logic for the Prodinno Diabetes Risk capstone.

Both the training scripts (train.py / retrain.py) and the live API import this
module, so the exact same cleaning rules are applied whether we are fitting a
model offline or scoring a single request online.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"

# These five columns use 0 as a disguised missing value in the raw Pima export
# (a body cannot have zero glucose, blood pressure, skin fold, insulin, or BMI).
SENTINEL_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, impute_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Replace sentinel zeros with NaN, then impute with the outcome-grouped median.

    If `impute_stats` is provided (as produced by a previous call), those medians
    are reused as-is instead of being recomputed — this is what the live API uses,
    so a single incoming record is imputed with the medians learned at training
    time rather than leaking information from itself.
    """
    out = df.copy()
    for col in SENTINEL_ZERO_COLUMNS:
        out.loc[out[col] == 0, col] = np.nan

    if impute_stats is None:
        impute_stats = {}
        group_col = TARGET_COLUMN if TARGET_COLUMN in out.columns else None
        for col in SENTINEL_ZERO_COLUMNS:
            if group_col:
                medians = out.groupby(group_col)[col].median().to_dict()
            else:
                medians = {}
            overall_median = float(out[col].median())
            impute_stats[col] = {"by_group": medians, "overall": overall_median}

    group_col = TARGET_COLUMN if TARGET_COLUMN in out.columns else None
    for col in SENTINEL_ZERO_COLUMNS:
        stats = impute_stats[col]
        if group_col:
            fill_by_group = out[group_col].map(stats["by_group"])
            out[col] = out[col].fillna(fill_by_group)
        out[col] = out[col].fillna(stats["overall"])

    return out, impute_stats


def get_feature_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int) if TARGET_COLUMN in df.columns else None
    return X, y


def clean_single_record(record: dict, impute_stats: dict) -> pd.DataFrame:
    """Clean one incoming API request the same way training data was cleaned."""
    row = {col: record.get(col, np.nan) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    for col in SENTINEL_ZERO_COLUMNS:
        if df.loc[0, col] == 0:
            df.loc[0, col] = np.nan
        if pd.isna(df.loc[0, col]):
            df.loc[0, col] = impute_stats[col]["overall"]
    return df.loc[:, FEATURE_COLUMNS].astype(float)
