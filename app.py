import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Food Adulteration Risk Prediction",
    layout="centered"
)

st.title("Food Adulteration Risk Prediction System")
st.write("Predict adulteration risk using a Machine Learning model.")

# -------------------------------
# Load model safely
# -------------------------------
MODEL_PATH = "food_adulteration_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -------------------------------
# User Inputs
# -------------------------------
food_type = st.selectbox(
    "Food Type",
    ["Milk", "Spices", "Oil", "Snacks"]
)

state = st.selectbox(
    "State",
    ["WB", "MH", "UP", "DL"]
)

sample_count = st.number_input(
    "Number of Samples",
    min_value=1,
    max_value=500,
    value=10
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "Food_Type": [food_type],
        "State": [state],
        "Sample_Count": [sample_count]
    })

    # One-hot encoding
    input_data = pd.get_dummies(input_data)

    # Align with training features
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )
    else:
        st.error("Model feature mismatch. Retrain the model.")
        st.stop()

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    if str(prediction).lower() == "high":
        st.error(f"Predicted Risk Level: {prediction}")
    else:
        st.success(f"Predicted Risk Level: {prediction}")

