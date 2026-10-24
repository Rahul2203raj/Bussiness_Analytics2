import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Medical Cost Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("Annual Medical Cost Predictor")
st.caption(
    "Estimate annual medical cost using a trained Linear Regression model. "
    "This is an estimate for planning and educational purposes, not medical, "
    "insurance, or financial advice."
)

MODEL_PATH = Path(__file__).resolve().parent / "model.sav"

FEATURES = [
    "previous_year_cost",
    "age",
    "bmi",
    "smoker",
    "hospital_admissions",
    "doctor_visits_per_year",
    "medication_count",
    "insurance_coverage_pct",
]

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Place model.sav in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load model.sav: {exc}")
    st.stop()

with st.form("cost_form"):
    st.subheader("Enter the required details")
    c1, c2 = st.columns(2)

    with c1:
        previous_year_cost = st.number_input(
            "Previous Year Medical Cost (₹)",
            min_value=0.0, value=10000.0, step=500.0
        )
        age = st.number_input(
            "Age (years)", min_value=0, max_value=120, value=40, step=1
        )
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=80.0, value=27.0, step=0.1
        )
        smoker = st.selectbox("Smoker", ["No", "Yes"])

    with c2:
        hospital_admissions = st.number_input(
            "Hospital Admissions (per year)",
            min_value=0, max_value=100, value=1, step=1
        )
        doctor_visits_per_year = st.number_input(
            "Doctor Visits (per year)",
            min_value=0, max_value=100, value=3, step=1
        )
        medication_count = st.number_input(
            "Medication Count", min_value=0, max_value=100, value=2, step=1
        )
        insurance_coverage_pct = st.slider(
            "Insurance Coverage (%)", min_value=0, max_value=100, value=70
        )

    submitted = st.form_submit_button(
        "Predict Annual Medical Cost", type="primary", use_container_width=True
    )

if submitted:
    input_data = {
        "previous_year_cost": previous_year_cost,
        "age": age,
        "bmi": bmi,
        "smoker": smoker,
        "hospital_admissions": hospital_admissions,
        "doctor_visits_per_year": doctor_visits_per_year,
        "medication_count": medication_count,
        "insurance_coverage_pct": insurance_coverage_pct,
    }
    row = pd.DataFrame([input_data], columns=FEATURES)

    try:
        # The saved model/pipeline should handle any required preprocessing.
        raw_prediction = float(model.predict(row)[0])

        # A linear regression can produce negative values; costs cannot.
        # Clip only the displayed prediction at zero.
        prediction = max(0.0, raw_prediction)

        st.metric("Estimated Annual Medical Cost", f"₹{prediction:,.2f}")

        if raw_prediction < 0:
            st.info(
                "The underlying model returned a negative estimate. "
                "Because medical costs cannot be negative, the displayed "
                "estimate has been floored at ₹0. This also indicates that "
                "the model may be unreliable for this input combination."
            )

        with st.expander("View input details"):
            st.dataframe(row, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that model.sav is compatible with "
            "these eight input features and that the feature names and "
            f"preprocessing match the model. Details: {exc}"
        )
