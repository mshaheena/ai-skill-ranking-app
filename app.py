import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 🚀 **App Title**
st.title("AI Skill Ranking Prediction App")

# 📌 **Load Dataset**
try:
    df = pd.read_csv("Coursera AI GSI Percentile and Category.csv")
    st.write("📂 Dataset Loaded Successfully!")
except FileNotFoundError:
    st.warning("⚠ Dataset not found. Please upload it to GitHub.")
except Exception as e:
    st.error(f"⚠ Error loading dataset: {e}")

# 📌 **Load Trained Model**
try:
    model = joblib.load("xgboost_ai_skill_model.pkl")
    st.write("✅ Model Loaded Successfully!")
except FileNotFoundError:
    st.warning("⚠ Model file not found. Please upload `xgboost_ai_skill_model.pkl` to GitHub.")

# 📊 **Dataset Overview**
st.subheader("📊 AI Skill Distribution by Region")
if 'df' in locals():
    fig, ax = plt.subplots(figsize=(8, 5))
    region_counts = df["region"].value_counts()
    sns.barplot(x=region_counts.index, y=region_counts.values, palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Region")
    plt.ylabel("Number of AI Professionals")
    plt.title("AI Skill Distribution by Region")
    st.pyplot(fig)
else:
    st.warning("⚠ Dataset not loaded. Please upload the dataset.")

# 🔥 **Heatmap - Correlation Matrix**
st.subheader("🔥 Correlation Heatmap of AI Skills")
if 'df' in locals():
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠ Dataset not loaded. Please upload the dataset.")

# 📦 **Boxplot - AI Skill Percentile by Region**
st.subheader("📦 AI Skill Percentile Distribution by Region")
if 'df' in locals():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="region", y="percentile_rank", data=df, palette="coolwarm", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Region")
    plt.ylabel("AI Skill Percentile Rank")
    plt.title("AI Skill Distribution by Region")
    st.pyplot(fig)
else:
    st.warning("⚠ Dataset not loaded. Please upload the dataset.")

# 🎯 **AI Skill Rank Prediction**
st.subheader("🎯 AI Skill Rank Prediction")

# 🌍 **User Inputs**
region = st.selectbox("🌍 Select Region", ["South America", "Asia", "North America"], key="region_select")
income_group = st.selectbox("💰 Select Income Group", ["Low", "Middle", "High"], key="income_select")
competency_id = st.slider("📈 Competency ID (Skill Level)", 0, 19, 5, key="competency_slider")

# ✅ **Feature Encoding (One-Hot Encoding)**
feature_vector = np.zeros(14)  # Ensure input shape is (1, 14)

# 🌍 **Encode Region**
region_mapping = {"South America": 4, "Asia": 3, "North America": 2}  # Adjust based on dataset encoding
if region in region_mapping:
    feature_vector[6 + region_mapping[region]] = 1  

# 💰 **Encode Income Group**
income_mapping = {"Low": 0, "Middle": 1, "High": 2}
if income_group in income_mapping:
    feature_vector[12 + income_mapping[income_group]] = 1  

# 🔢 **Set Competency ID**
feature_vector[3] = competency_id  

# Convert to NumPy array for prediction
user_input = np.array([feature_vector])

# Debugging Output
st.write(f"✅ **Input Shape:** {user_input.shape}")  # Should be (1, 14)
st.write(f"✅ **Feature Vector:** {user_input}")  # Debugging input values

# 🎯 **Prediction Button**
if st.button("Predict AI Skill Rank", key="predict_button"):
    try:
        prediction = model.predict(user_input)[0]
        st.success(f"🎯 Predicted AI Skill Rank: {prediction:.2f}")
    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")

# ✅ **Debugging Button**
if st.button("Run Code", key="run_code_button"):
    st.write("✅ **The code ran successfully!**")
