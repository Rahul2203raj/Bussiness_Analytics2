import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Medical Cost Predictor", page_icon="🩺", layout="centered")
st.title("Annual Medical Cost Predictor")
st.caption("Estimate annual medical cost using a Linear Regression model. This is an estimate, not medical or financial advice.")

@st.cache_resource
def load_model():
    return joblib.load("medical_cost_model.sav")

model=load_model()
with st.form("cost_form"):
    c1,c2=st.columns(2)
    with c1:
        previous_year_cost=st.number_input("Previous Year Medical Cost",min_value=0,value=10000,step=500)
        age=st.number_input("Age",min_value=0,max_value=120,value=40,step=1)
        bmi=st.number_input("BMI",min_value=10.0,max_value=80.0,value=27.0,step=0.1)
        smoker=st.selectbox("Smoker",["No","Yes"])
    with c2:
        hospital_admissions=st.number_input("Hospital Admissions (per year)",min_value=0,max_value=100,value=1,step=1)
        doctor_visits_per_year=st.number_input("Doctor Visits (per year)",min_value=0,max_value=100,value=3,step=1)
        medication_count=st.number_input("Medication Count",min_value=0,max_value=100,value=2,step=1)
        insurance_coverage_pct=st.slider("Insurance Coverage (%)",min_value=0,max_value=100,value=70)
    submit=st.form_submit_button("Predict Annual Medical Cost",type="primary")

if submit:
    row=pd.DataFrame([{
      "previous_year_cost":previous_year_cost,"age":age,"bmi":bmi,"smoker":smoker,
      "hospital_admissions":hospital_admissions,"doctor_visits_per_year":doctor_visits_per_year,
      "medication_count":medication_count,"insurance_coverage_pct":insurance_coverage_pct
    }],columns=model.feature_names_in_)
    
prediction = float(model.predict(row)[0])

# Prevent negative medical cost predictions
prediction = max(0, prediction)

st.metric(
    "Estimated Annual Medical Cost",
    f"₹{prediction:,.2f}"
)
