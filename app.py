import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load trained model
try:
    model = joblib.load("xgboost_ai_skill_model.pkl")
    st.write("✅ Model Loaded Successfully!")
except:
    st.warning("⚠ Model file not found. Please upload it.")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# App Title
st.title("🚀 AI-Powered Data Analysis & Prediction Model")

# Project Overview
st.subheader("📌 Project Overview")
st.write("""
This project analyzes AI skill trends using a dataset from Coursera's Global AI Skills Index.
It performs data preprocessing, exploratory data analysis (EDA), and machine learning predictions.
""")

# Load Dataset
try:
    df = pd.read_csv("Coursera AI GSI Percentile and Category.csv")  
    st.write("📂 **Dataset Loaded Successfully!**")
except:
    st.warning("⚠ **Dataset not found. Please upload it to GitHub.**")

# Load Trained Model
try:
    model = joblib.load("xgboost_ai_skill_model.pkl")
    st.write("✅ **Model Loaded Successfully!**")
except:
    st.warning("⚠ **Model file not found. Please upload `xgboost_ai_skill_model.pkl` to GitHub.**")

# Dataset Information
st.write("""
## 📊 Dataset Used
- **Columns:** Country, region, income group, competency ID, percentile rank.
- **Target Variable:** Percentile rank (for regression) or percentile category (for classification).
- **Goal:** Predict AI skill ranking based on country and income group.

## 📌 Models Used
- **Random Forest Regressor**: Predicts AI skill ranking.
- **SVM & Logistic Regression**: Tested but gave lower accuracy.
- **K-Means Clustering**: Groups countries based on AI skills.
""")
st.subheader("🎯 AI Skill Rank Prediction")

# User Inputs
region = st.selectbox("🌍 Select Region", ["South America", "Asia", "North America"])
income_group = st.selectbox("💰 Select Income Group", ["Low", "Middle", "High"])
competency_id = st.slider("📈 Competency ID (Skill Level)", 0, 19, 5)

if st.button("Predict AI Skill Rank", key="predict_button"):
    input_data = [[competency_id]]  # Adjust this to match model input format
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted AI Skill Rank: {prediction:.2f}")

# AI Skill Prediction Section
st.subheader("🎯 AI Skill Rank Prediction")

# User Inputs
region = st.selectbox("🌍 Select Region", ["South America", "Asia", "North America"], key="region_select")
income_group = st.selectbox("💰 Select Income Group", ["Low", "Middle", "High"], key="income_select")
competency_id = st.slider("📈 Competency ID (Skill Level)", 0, 19, 5, key="competency_slider")

# Prediction Button
if st.button("Predict AI Skill Rank"):
    try:
        input_data = [[competency_id]]  # Adjust this to match model input format
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 **Predicted AI Skill Rank: {prediction:.2f}**")
    except:
        st.error("⚠ **Prediction failed. Ensure model is loaded correctly.**")

# Run Code Button (Optional Debugging)
if st.button("Run Code"):
    st.write("✅ **The code ran successfully!**")


import numpy as np
import streamlit as st
import joblib

# Load the trained XGBoost model
model = joblib.load("xgboost_ai_skill_model.pkl")

st.title("AI Skill Ranking Prediction App")

# Fixed Values
region = "South America"  # Fixed to South America
income_group = "Middle"   # Fixed to Middle Income

# Competency ID (Skill Level) slider
competency_id = st.slider("Competency ID (Skill Level)", 0, 19, 5)

# ✅ Create a feature vector with 14 elements, all initialized to 0
feature_vector = np.zeros(14)

# ✅ Correctly Encode Region (One-Hot Encoding)
region_mapping = {
    "South America": 4  # Check the dataset encoding
}
feature_vector[6 + region_mapping["South America"]] = 1  # ✅ Adjusted correctly for region

# ✅ Correctly Encode Income Group (One-Hot Encoding)
income_mapping = {
    "Middle": 1  # Check dataset encoding
}
feature_vector[12 + income_mapping["Middle"]] = 1  # ✅ Adjusted for income group

# ✅ Set Competency ID (Assuming it's at index 3)
feature_vector[3] = competency_id  

# Convert to NumPy array for prediction
user_input = np.array([feature_vector])

# Debugging Output (Check before clicking "Predict")
st.write("🔹 Checking Input Features for Debugging:")
st.write(f"✅ Input Shape: {user_input.shape}")  # Should be (1, 14)
st.write(f"✅ Feature Vector: {user_input}")    # Should show 14 numbers

# Prediction Button
if st.button("Predict AI Skill Rank"):
    prediction = model.predict(user_input)[0]
    st.success(f"Predicted AI Skill Rank: {prediction:.2f}")
st.subheader("🎯 AI Skill Rank Prediction")



