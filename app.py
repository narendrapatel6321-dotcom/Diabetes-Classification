import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "risk_score" not in st.session_state:
    st.session_state.risk_score = None

if "diabetes_pred" not in st.session_state:
    st.session_state.diabetes_pred = None

if "diabetes_prob" not in st.session_state:
    st.session_state.diabetes_prob = None

# --------------------------------------------------
# Load models (cached)
# --------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "risk_model": joblib.load("models/risk_score_model.pkl"),
        "binary_model": joblib.load("models/diagnosed_diabetes_model.pkl"),
        "type_model": joblib.load("models/diabetes_type_model.pkl"),
    }

models = load_models()
risk_model = models["risk_model"]
binary_model = models["binary_model"]
type_model = models["type_model"]

# --------------------------------------------------
# UI Header
# --------------------------------------------------
st.title("🩺 Diabetes Prediction System")
st.write("Multi-stage machine learning based diabetes prediction")

# ==================================================
# STAGE 1 — Risk Score
# ==================================================
st.header("Stage 1: Diabetes Risk Score")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", 18, 100, 35,
        help="Enter age in completed years (18–100)."
    )
with col2:
    gender = st.selectbox(
        "Gender", ["Male", "Female", "Other"],
        help="Biological sex of the individual."
    )

col3 , col4 = st.columns(2)
with col3:
    ethnicity = st.selectbox(
        "Ethnicity", ["Asian", "Black", "Hispanic", "White", "Other"],
        help="Self-reported ethnic background."
    )
with col4:
    education_level = st.selectbox(
        "Education Level",
        ["No formal", "Highschool", "Graduate", "Postgraduate"],
        help="Highest level of education completed."
    )

col5, col6 = st.columns(2)
with col5:
    income_level = st.selectbox(
        "Income Level",
        ["Low", "Lower-Middle", "Medium", "Upper-Middle", "High"],
        help="Approximate household income category."
    )
with col6:
    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Unemployed", "Student", "Retired"],
        help="Current employment situation."
    )

col7, col8= st.columns(2)
with col7:
    smoking_status = st.selectbox(
        "Smoking Status",
        ["Never", "Former", "Current"],
        help="Smoking history."
    )
with col8:
    alcohol_consumption_per_week = st.number_input(
        "Alcohol Consumption / Week", 0, 10, 0,
        help="Number of alcoholic drinks per week."
    )

col9,col10 = st.columns(2)
with col9:
    physical_activity_minutes_per_week = st.number_input(
        "Physical Activity (min/week)", 0, 800, 0,
        help="Total minutes of physical activity per week."
    )

with col10:
    diet_score = st.number_input(
        "Diet Score (0–10)", 0, 10, 5,
        help="Higher score indicates healthier diet."
    )

col11, col12 = st.columns(2)
with col11:
    sleep_hours_per_day = st.number_input(
        "Sleep Hours / Day", 0, 12, 7,
        help="Average hours of sleep per night."
    )
with col12:
    screen_time_hours_per_day = st.number_input(
        "Screen Time (hrs/day)", 0, 16, 4,
        help="Average daily screen exposure."
    )

col13, col14 = st.columns(2)
with col13:
    family_history = st.selectbox(
        "Family History of Diabetes", ["Yes", "No"],
        help="Any first-degree relative with diabetes?"
    )
with col14:
    hypertension_history = st.selectbox(
        "Hypertension History", ["Yes", "No"],
        help="History of high blood pressure."
    )

col15,col16 = st.columns(2)
with col15:
    cardiovascular_history = st.selectbox(
        "Cardiovascular History", ["Yes", "No"],
        help="History of heart-related disease."
    )
with col16:
    bmi = st.number_input(
        "BMI", 10.0, 60.0, 25.0,
        help="Body Mass Index."
    )

col17, col18 = st.columns(2)
with col17:
    waist_to_hip_ratio = st.number_input(
        "Waist-to-Hip Ratio", 0.5, 1.5, 0.9,
        help="Indicator of fat distribution."
    )
with col18:
    heart_rate = st.number_input(
        "Heart Rate", 40, 150, 70,
        help="Resting heart rate (bpm)."
    )

col19, col20 = st.columns(2)
with col19:
    systolic_bp = st.number_input(
        "Systolic BP", 80, 200, 120,
        help="Upper blood pressure value."
    )
with col20:
    diastolic_bp = st.number_input(
        "Diastolic BP", 60, 130, 80,
        help="Lower blood pressure value."
    )

# Binary encoding
family_history = 1 if family_history == "Yes" else 0
hypertension_history = 1 if hypertension_history == "Yes" else 0
cardiovascular_history = 1 if cardiovascular_history == "Yes" else 0

# Base feature dictionary
base_features = {
    "age": age,
    "gender": gender,
    "ethnicity": ethnicity,
    "education_level": education_level,
    "income_level": income_level,
    "employment_status": employment_status,
    "smoking_status": smoking_status,
    "alcohol_consumption_per_week": alcohol_consumption_per_week,
    "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
    "diet_score": diet_score,
    "sleep_hours_per_day": sleep_hours_per_day,
    "screen_time_hours_per_day": screen_time_hours_per_day,
    "family_history_diabetes": family_history,
    "hypertension_history": hypertension_history,
    "cardiovascular_history": cardiovascular_history,
    "bmi": bmi,
    "waist_to_hip_ratio": waist_to_hip_ratio,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate,
}

if st.button("Predict Risk Score"):
    risk_df = pd.DataFrame([base_features])
    st.session_state.risk_score = risk_model.predict(risk_df)[0]
    st.session_state.diabetes_pred = None
    st.session_state.diabetes_prob = None

if st.session_state.risk_score is not None:
    st.success(f"Predicted Risk Score: {st.session_state.risk_score:.2f}")

# ==================================================
# STAGE 2 — Diabetes Prediction
# ==================================================
if st.session_state.risk_score is not None:

    st.header("Stage 2: Diabetes Prediction")

    col21, col22= st.columns(2)
    with col21:
        cholesterol_total = st.number_input(
        "Total Cholesterol (mg/dL)", 100, 400, 200,
        help="Total blood cholesterol level."
    )
    with col22:
        hdl_cholesterol = st.number_input(
        "HDL Cholesterol (mg/dL)", 20, 100, 50,
        help="Good cholesterol."
    )
        
    col23,col24 = st.columns(2)
    with col23:
        ldl_cholesterol = st.number_input(
        "LDL Cholesterol (mg/dL)", 50, 300, 100,
        help="Bad cholesterol."
    )
    with col24:
        triglycerides = st.number_input(
        "Triglycerides (mg/dL)", 50, 500, 150,
        help="Fat content in blood."
    )
        
    col25 , col26,   = st.columns(2)
    with col25 :
        glucose_fasting = st.number_input(
        "Fasting Glucose (mg/dL)", 70, 200, 90,
        help="Blood glucose after fasting."
    )
    with col26:
        glucose_postprandial = st.number_input(
        "Post-meal Glucose (mg/dL)", 70, 300, 120,
        help="Blood glucose after meals."
    )
    
    col27 , col28   = st.columns(2)
    with col27 :
        insulin_level = st.number_input(
        "Insulin Level (µIU/mL)", 2, 50, 10,
        help="Circulating insulin level."
    )
    with col28:
        hba1c = st.number_input(
        "HbA1c (%)", 4.0, 14.0, 5.5,
        help="Average blood sugar over 2–3 months."
    )
    diabetes_features = {
            **base_features,
            "cholesterol_total": cholesterol_total,
            "hdl_cholesterol": hdl_cholesterol,
            "ldl_cholesterol": ldl_cholesterol,
            "triglycerides": triglycerides,
            "glucose_fasting": glucose_fasting,
            "glucose_postprandial": glucose_postprandial,
            "insulin_level": insulin_level,
            "hba1c": hba1c,
            "diabetes_risk_score_pred": st.session_state.risk_score,
        }

    if st.button("Predict Diabetes"):
      
        diabetes_df = pd.DataFrame([diabetes_features])
        st.session_state.diabetes_pred = binary_model.predict(diabetes_df)[0]
        st.session_state.diabetes_prob = binary_model.predict_proba(diabetes_df)[0, 1]

    if st.session_state.diabetes_pred is not None:
        if st.session_state.diabetes_pred == 1:
            st.error(
                f"Diabetes Detected (Probability: {st.session_state.diabetes_prob:.2%})"
            )
        else:
            st.success("No Diabetes Detected")

# ==================================================
# STAGE 3 — Diabetes Type Prediction
# ==================================================
if st.session_state.diabetes_pred == 1:

    st.header("Stage 3: Diabetes Type")

    if st.button("Predict Diabetes Type"):
        type_features = {
            **diabetes_features,
            "diagnosed_diabetes_prob": st.session_state.diabetes_prob,
        }

        type_df = pd.DataFrame([type_features])
        diabetes_type = type_model.predict(type_df)[0]
        if diabetes_type == 1:
            diabetes_type = "Type 1 Diabetes"
        elif diabetes_type == 2:
            diabetes_type = "Type 2 Diabetes"
        else:
            diabetes_type = "Gestational Diabetes"
        st.info(f"Predicted Diabetes Type: {diabetes_type}")
