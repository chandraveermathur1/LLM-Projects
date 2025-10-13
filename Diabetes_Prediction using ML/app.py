import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# App title
st.title("🩺 Diabetes Risk Score Predictor")
st.markdown("Enter your health parameters below:")

# Input fields
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
blood_glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=300.0, value=110.0)
physical_activity = st.number_input("Physical Activity (mins/day)", min_value=0.0, max_value=300.0, value=30.0)

# Updated UI with user-friendly options
diet_option = st.selectbox("Healthy Diet?", ["Yes", "No"])
diet = 1 if diet_option == "Yes" else 0

med_adherence_option = st.selectbox("Medication Adherence", ["Poor", "Good", "Excellent"])
medication_adherence = {"Poor": 0, "Good": 1, "Excellent": 2}[med_adherence_option]

stress_option = st.selectbox("High Stress Level?", ["Yes", "No"])
stress_level = 1 if stress_option == "Yes" else 0

hydration_option = st.selectbox("Hydration Adequate?", ["Yes", "No"])
hydration_level = 1 if hydration_option == "Yes" else 0

sleep_hours = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)

# Calculate BMI dynamically
height_m = height / 100  # convert cm to meters
bmi = round(weight / (height_m ** 2), 2)

# Show calculated BMI
st.markdown(f"🧮 **Calculated BMI:** {bmi}")

# Prediction
if st.button("Predict Risk Score"):
    input_df = pd.DataFrame([{
        "weight": weight,
        "height": height,
        "blood_glucose": blood_glucose,
        "physical_activity": physical_activity,
        "diet": diet,
        "medication_adherence": medication_adherence,
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "hydration_level": hydration_level,
        "bmi": bmi
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Diabetes Risk Score: **{prediction:.2f}**")
