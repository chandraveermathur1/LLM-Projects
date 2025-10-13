import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")

# Input features
age_days = st.number_input("Age (in days)", min_value=10000, max_value=30000, value=20000)
gender = st.selectbox("Gender", options=[1, 2])  # 1: women, 2: men
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
cholesterol = st.selectbox("Cholesterol", options=[1, 2, 3])  # 1: normal, 2: above, 3: well above
gluc = st.selectbox("Glucose", options=[1, 2, 3])
smoke = st.selectbox("Do you smoke?", options=[0, 1])
alco = st.selectbox("Do you drink alcohol?", options=[0, 1])
active = st.selectbox("Are you physically active?", options=[0, 1])

# Convert age to years
age_years = age_days // 365

# Match the exact column names and order from training
feature_names = ['gender', 'height', 'weight', 'age_years',
                 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Create DataFrame from input
features_df = pd.DataFrame([[gender, height, weight, age_years,
                             ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]],
                           columns=feature_names)

# Scale input features
features_scaled = scaler.transform(features_df)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(features_scaled)
    result = "Cardiovascular Disease" if prediction[0] == 1 else "No Disease"
    st.success(f"Prediction: {result}")
