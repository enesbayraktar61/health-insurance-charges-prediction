import os
import json
import joblib
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="Insurance Charges Prediction", layout="centered")

st.title("Health Insurance Charges Prediction")
st.write("Enter your information to predict estimated insurance charges.")

# Base directory (Hugging Face repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (files are stored in the repo root)
MODEL_PATH = os.path.join(BASE_DIR, "insurance_charges_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "training_columns.json")

# Load model and training columns
model = joblib.load(MODEL_PATH)

with open(COLUMNS_PATH, "r") as f:
    training_columns = json.load(f)

st.subheader("Input Features")

# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create full input row with all required columns
input_data = {col: 0 for col in training_columns}

# Fill known inputs
input_data.update({
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
})

# Convert to DataFrame with correct column order
input_df = pd.DataFrame([input_data], columns=training_columns)

if st.button("Predict Charges"):
    # Predict using the trained pipeline
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Insurance Charges: ${pred:,.2f}")

st.caption("Model: RandomForestRegressor + sklearn Pipeline (preprocessing included).")
