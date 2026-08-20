"""
Streamlit front-end for the Prodinno Diabetes Risk capstone.

Talks to the FastAPI service (api/main.py) over plain HTTP — this file never
touches the model directly, exactly like a real production UI wouldn't.
Set API_URL to point elsewhere; defaults to the docker-compose service name.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Prodinno Diabetes Risk", page_icon="🩺", layout="wide")

st.image("assets/prodinno_logo.png", width=220)
st.title("Diabetes Risk — Capstone Demo")
st.caption(
    "Session recap → production: the same Pima Indians Diabetes dataset from the "
    "XGBoost notebook, now served behind a versioned, explainable API."
)

with st.sidebar:
    st.header("Patient inputs")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=25.0)
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

    st.caption(
        "Leaving a field at a biologically-impossible 0 (e.g. Glucose) is fine — "
        "the API imputes disguised missing values exactly like the XGBoost "
        "notebook's data-impurity lesson."
    )

features = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}

try:
    versions_resp = requests.get(f"{API_URL}/versions", timeout=10)
    versions_resp.raise_for_status()
    registry = versions_resp.json()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API at {API_URL}: {exc}")
    st.stop()

version_options = [v["version"] for v in registry["versions"]][::-1]
selected_version = st.selectbox(
    "Model version",
    options=version_options,
    index=0,
    help="Every retrain adds a new timestamped version here — nothing is overwritten.",
)

col_predict, col_retrain = st.columns([1, 1])

with col_predict:
    if st.button("Predict", type="primary", use_container_width=True):
        resp = requests.post(
            f"{API_URL}/predict",
            json={"features": features, "version": selected_version},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        st.session_state["last_prediction"] = result
        st.session_state["last_features"] = features
        st.session_state["last_version"] = selected_version

with col_retrain:
    if st.button("Retrain model", use_container_width=True):
        with st.spinner("Retraining on the full dataset and registering a new version..."):
            resp = requests.post(f"{API_URL}/retrain", json={"note": "triggered from Streamlit UI"}, timeout=120)
            resp.raise_for_status()
            new_version = resp.json()
        st.success(f"New version registered: {new_version['version']}")
        st.rerun()

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    proba = result["probability_diabetes"]
    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model version", result["version"])
    m2.metric("Diabetes probability", f"{proba:.1%}")
    m3.metric("Label", result["label"].replace("_", " ").title())
    st.progress(min(max(proba, 0.0), 1.0))

    if st.button("Explain this prediction (SHAP + LIME)"):
        resp = requests.post(
            f"{API_URL}/explain",
            json={"features": st.session_state["last_features"], "version": st.session_state["last_version"]},
            timeout=30,
        )
        resp.raise_for_status()
        explanation = resp.json()

        shap_col, lime_col = st.columns(2)
        with shap_col:
            st.markdown("**SHAP — Shapley value contributions**")
            shap_df = pd.DataFrame(
                sorted(explanation["shap"]["contributions"].items(), key=lambda kv: abs(kv[1]), reverse=True),
                columns=["feature", "shap_value"],
            ).set_index("feature")
            st.bar_chart(shap_df)
            st.caption(
                f"Base value (average model output): {explanation['shap']['base_value']:.3f}. "
                "Positive bars push the prediction toward 'diabetes', negative bars pull it away."
            )
        with lime_col:
            st.markdown("**LIME — local surrogate explanation**")
            lime_df = pd.DataFrame(explanation["lime"]["contributions"]).set_index("rule")
            st.bar_chart(lime_df)
            st.caption(
                "Each rule is a condition on one feature in this patient's local neighborhood; "
                "the weight is that condition's effect in LIME's local linear model."
            )

st.divider()
st.subheader("Version history")
history_df = pd.DataFrame(
    [
        {
            "version": v["version"],
            "timestamp": v["timestamp"],
            "note": v.get("note", ""),
            **v["metrics"],
        }
        for v in registry["versions"]
    ]
)
st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
