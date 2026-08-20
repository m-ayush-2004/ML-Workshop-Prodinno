<p align="center">
  <img src="assets/prodinno_logo.png" alt="Prodinno" width="300">
</p>

<h1 align="center">Capstone — Diabetes Risk, Productionized</h1>

<p align="center"><i>Same dataset as the XGBoost session, now trained, served, versioned,
retrained, and explained behind a real API and UI.</i></p>

---

## What this is

The `04_xgboost/` notebooks trained a diabetes-risk classifier once, inside a notebook,
by hand. This capstone takes that same problem and answers the next question every ML
team eventually asks: **how does this become a system someone can actually use, retrain,
and trust?**

- **Training** (`src/train.py`) — cleans the data exactly like the XGBoost notebook did
  (disguised missing zeros → NaN → outcome-grouped median imputation), fits an XGBoost
  classifier, evaluates it, and saves it as a new model version.
- **Model versioning** — every training run (including the very first one, run
  automatically on startup if none exists) is saved to `models/v_<YYYYMMDD_HHMMSS>/` with
  its own `model.joblib`, `metrics.json`, and imputation statistics. Nothing is ever
  overwritten; `models/registry.json` tracks every version and which one is `"latest"`.
- **Retraining** (`src/retrain.py`) — re-runs training (optionally with new rows appended)
  and registers another new version, so you always have a full audit trail of every model
  that has ever been in production.
- **Explainability** (`src/explain.py`) — SHAP (exact Shapley values from the trees) and
  LIME (local linear surrogate) for any single prediction. See
  `notebooks/00_capstone_walkthrough.ipynb` for the full theory.
- **API** (`api/main.py`, FastAPI) — `GET /versions`, `POST /predict`, `POST /explain`,
  `POST /retrain`.
- **UI** (`ui/app.py`, Streamlit) — a patient-input form, a live prediction, a
  SHAP + LIME explanation panel, a version-history table, and a "Retrain" button.

## Running it

### With Docker (recommended — this is the "proper setup")

```powershell
cd 06_capstone
docker compose up --build
```

- API → http://localhost:8000/docs (interactive Swagger UI)
- UI → http://localhost:8501

The first time the API container starts, it finds no `models/registry.json`, so it trains
and registers the first model version automatically — no manual step required.

### Without Docker (using the repo's shared venv)

From the repo root, after running `.\setup.ps1` once:

```powershell
.\.venv\Scripts\Activate.ps1
cd 06_capstone
python src/train.py
uvicorn api.main:app --reload --port 8000        # terminal 1
streamlit run ui/app.py                           # terminal 2, from 06_capstone/
```

## Model registry layout

```
models/
├── registry.json                # {"versions": [...], "latest": "v_20260101_100000"}
├── v_20260101_100000/
│   ├── model.joblib
│   ├── metrics.json
│   ├── impute_stats.json
│   ├── feature_columns.json
│   └── background_sample.csv    # LIME's local-neighborhood background sample
└── v_20260101_113000/            # a later retrain — the previous version is untouched
    └── ...
```

## API quick reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/versions` | — | full registry |
| POST | `/predict` | `{"features": {...}, "version": "v_... (optional)"}` | probability + label |
| POST | `/explain` | `{"features": {...}, "version": "optional", "method": "shap"/"lime"/"both"}` | SHAP + LIME contributions |
| POST | `/retrain` | `{"note": "optional"}` | the newly registered version + its metrics |
