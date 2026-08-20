"""
Explainability helpers for the diabetes-risk model — SHAP (global + local,
game-theoretic Shapley values) and LIME (local linear surrogate).

Both explainers answer "why did the model say this?" for one specific
prediction, but they get there very differently:

- SHAP's TreeExplainer walks the XGBoost trees directly and computes exact
  Shapley values in polynomial time — each feature's contribution is the
  average marginal effect of adding it across every possible ordering of
  features, so contributions for a row always sum exactly to
  (prediction - base_value).
- LIME instead perturbs the input many times, asks the real model for a
  prediction on each perturbed sample, and fits a small interpretable linear
  model to that local neighborhood — it explains any black-box model, not
  just trees, at the cost of being an approximation that can vary run to run.

See `06_capstone/notebooks/00_capstone_walkthrough.ipynb` for the full theory
(Shapley value formula, LIME's weighted-regression objective) and plots.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from data_processing import FEATURE_COLUMNS


def load_version_artifacts(version_dir: Path) -> dict:
    return {
        "model": joblib.load(version_dir / "model.joblib"),
        "background": pd.read_csv(version_dir / "background_sample.csv"),
    }


def shap_explain(model, _background: pd.DataFrame, row: pd.DataFrame) -> dict:
    # TreeExplainer computes exact Shapley values from the tree structure itself,
    # so unlike LIME it needs no background sample — kept as a parameter only to
    # give both explainers the same call signature.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    values = np.asarray(shap_values).reshape(-1)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).reshape(-1)[0])
    return {
        "base_value": float(base_value),
        "contributions": {
            col: float(val) for col, val in zip(FEATURE_COLUMNS, values)
        },
    }


def lime_explain(model, background: pd.DataFrame, row: pd.DataFrame, num_features: int = 8) -> dict:
    explainer = LimeTabularExplainer(
        training_data=background[FEATURE_COLUMNS].values,
        feature_names=FEATURE_COLUMNS,
        class_names=["no_diabetes", "diabetes"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(x, columns=FEATURE_COLUMNS))

    explanation = explainer.explain_instance(
        row[FEATURE_COLUMNS].values[0],
        predict_fn,
        num_features=num_features,
    )
    return {
        "contributions": [
            {"rule": rule, "weight": float(weight)}
            for rule, weight in explanation.as_list()
        ]
    }
