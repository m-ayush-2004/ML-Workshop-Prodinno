<p align="center">
  <img src="assets/prodinno_logo.png" alt="Prodinno" width="360">
</p>

<h1 align="center">Prodinno — One-Day Machine Learning Workshop</h1>

<p align="center"><i>From raw, imperfect data to a deployed, explainable model — in one day.</i></p>

---

## Event Flow

| Time | Session |
|---|---|
| 10:00 AM | Welcome & Kickoff — introductions, agenda, Kaggle/Colab setup check |
| 10:15 AM | **Session 1** — Linear & Logistic Regression |
| 10:55 AM | **Session 2** — XGBoost & Random Forest |
| 11:35 AM | **Session 3** — Hierarchical Clustering & Dendrogram |
| 12:00 PM | Lunch Break |
| 1:00 PM | Dataset Briefing |
| 1:15 PM | Hands-On Practical Session (Regression → RF vs. XGBoost → Clustering) |
| 3:00 PM | Notebook Finalization |
| 3:15 PM | Speaker Evaluation |
| 3:45 PM | Closing & Takeaways |
| 4:00 PM | End |

## What's in this repo

Six folders, each a self-contained topic. Every topic folder follows the **same four-notebook
flow**, so once you've worked through one, the rest feel familiar:

| Order | Notebook | What happens |
|---|---|---|
| `00_dataset_and_impurity.ipynb` | Load the single real dataset, deliberately look for/introduce **one** realistic data-quality issue, detect it properly, and fix it *without* damaging the signal in the data. |
| `01_eda.ipynb` | Explore the (now clean) data with matplotlib, seaborn, and plotly — because **the data drives the model, not the other way around.** |
| `02_data_processing.ipynb` | Encode, scale, engineer features, and split — turning tidy data into model-ready arrays. |
| `03_train_test_eval.ipynb` | Train the algorithm, evaluate it properly, and go deep on the math: what the loss function is, how the algorithm actually works, how to read the results, and the wider landscape of related loss/objective functions. |

| Folder | Topic | Dataset | Data-quality lesson |
|---|---|---|---|
| `01_linear_regression/` | Linear Regression | Avocado Prices (Hass Avocado Board) | Outliers (IQR / Z-score) |
| `02_logistic_regression/` | Logistic Regression | Breast Cancer Wisconsin | Class imbalance (SMOTE / class weights) |
| `03_random_forest/` | Random Forest | Hotel Booking Demand | Duplicate records |
| `04_xgboost/` | XGBoost | Pima Indians Diabetes | Disguised missing values (sentinel zeros) |
| `05_hierarchical_clustering/` | Hierarchical Clustering | Country Indicators | Inconsistent categorical text |
| `06_capstone/` | Capstone | Pima Indians Diabetes | End-to-end: Docker, UI, retraining, model versioning, SHAP & LIME |

## Setup (Windows Terminal / PowerShell)

From the repo root:

```powershell
.\setup.ps1
```

This will:
1. Create a local virtual environment in `.venv`
2. Install every dependency declared in `pyproject.toml` (pandas, scikit-learn, xgboost,
   matplotlib, seaborn, plotly, shap, lime, streamlit, fastapi, jupyter, …)
3. Register a Jupyter kernel named **"Prodinno ML Workshop"**

Then open any notebook (VS Code, JupyterLab, or `jupyter notebook`) and select the
**Prodinno ML Workshop** kernel.

To come back to the environment in a new terminal later:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Recommended run order

Work through the folders in numeric order (`01_` → `05_`) — each mirrors the event flow
above. The capstone (`06_capstone/`) is the closing project: read its own `README.md` for
the Docker instructions.
