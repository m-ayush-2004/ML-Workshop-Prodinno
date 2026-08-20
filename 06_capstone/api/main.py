"""
FastAPI model-serving layer for the Prodinno Diabetes Risk capstone.

Endpoints
---------
GET  /health    -> liveness check
GET  /versions  -> full model registry (every trained version + which is "latest")
POST /predict   -> run inference with a given (or latest) model version
POST /explain   -> SHAP + LIME explanation for one prediction
POST /retrain   -> train a brand-new timestamped version, register it, keep all prior ones
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data_processing import clean_single_record  # noqa: E402
from explain import lime_explain, load_version_artifacts, shap_explain  # noqa: E402
from retrain import retrain as retrain_model  # noqa: E402
from train import ensure_initial_model  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_initial_model(now=now)
    yield


app = FastAPI(title="Prodinno Diabetes Risk API", version="1.0.0", lifespan=lifespan)


class Features(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class PredictRequest(BaseModel):
    features: Features
    version: str | None = None


class ExplainRequest(BaseModel):
    features: Features
    version: str | None = None
    method: str = "both"  # "shap" | "lime" | "both"


class RetrainRequest(BaseModel):
    note: str | None = None


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise HTTPException(status_code=503, detail="No model has been trained yet.")
    return json.loads(REGISTRY_PATH.read_text())


def _resolve_version(version: str | None) -> str:
    registry = _load_registry()
    if version is None:
        version = registry["latest"]
    if version is None:
        raise HTTPException(status_code=503, detail="No model has been trained yet.")
    known = {v["version"] for v in registry["versions"]}
    if version not in known:
        raise HTTPException(status_code=404, detail=f"Unknown model version '{version}'.")
    return version


@lru_cache(maxsize=16)
def _load_artifacts(version: str):
    version_dir = MODELS_DIR / version
    artifacts = load_version_artifacts(version_dir)
    impute_stats = json.loads((version_dir / "impute_stats.json").read_text())
    artifacts["impute_stats"] = impute_stats
    return artifacts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/versions")
def versions():
    return _load_registry()


@app.post("/predict")
def predict(req: PredictRequest):
    version = _resolve_version(req.version)
    artifacts = _load_artifacts(version)
    row = clean_single_record(req.features.model_dump(), artifacts["impute_stats"])
    proba = float(artifacts["model"].predict_proba(row)[0, 1])
    return {
        "version": version,
        "probability_diabetes": round(proba, 4),
        "prediction": int(proba >= 0.5),
        "label": "diabetes" if proba >= 0.5 else "no_diabetes",
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    version = _resolve_version(req.version)
    artifacts = _load_artifacts(version)
    row = clean_single_record(req.features.model_dump(), artifacts["impute_stats"])

    result = {"version": version}
    if req.method in ("shap", "both"):
        result["shap"] = shap_explain(artifacts["model"], artifacts["background"], row)
    if req.method in ("lime", "both"):
        result["lime"] = lime_explain(artifacts["model"], artifacts["background"], row)
    return result


@app.post("/retrain")
def retrain(req: RetrainRequest):
    result = retrain_model(note=req.note or "triggered via API")
    _load_artifacts.cache_clear()
    return result
