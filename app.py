# app.py
import pickle
import streamlit as st
import numpy as np

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below to check the diabetes risk:")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 110)
bp = st.number_input("Blood Pressure", 0, 120, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    if prob > 0.6:  # threshold set to 0.6 for better balance
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")
        import os, pickle

MODEL_PATH = "diabetes_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please check deployment.")
else:
    model = pickle.load(open(MODEL_PATH, "rb"))


