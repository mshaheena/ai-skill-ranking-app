import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBRegressor

# 🚀 **App Title**
st.title("AI Skill Ranking Prediction App")

# 📌 **Load Dataset**
try:
    df = pd.read_csv("Coursera AI GSI Percentile and Category.csv")
    df['competency_id'] = pd.to_numeric(df['competency_id'], errors='coerce')  # Convert to numeric
    df.dropna(inplace=True)  # Remove missing values
    st.write("📂 Dataset Loaded Successfully!")
except FileNotFoundError:
    st.error("⚠ Dataset not found. Please upload it to GitHub.")
    df = None

# 📌 **Dataset Health Check**
if df is not None:
    st.subheader("📌 Checking Dataset Health")

    # **Check if dataset is empty**
    if df.empty:
        st.error("❌ Dataset is empty! Please check your CSV file.")

    # **Check for missing values**
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning(f"⚠ Warning: Missing values detected:\n{missing_values}")

    # **Ensure numeric columns exist**
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.error("⚠ No numeric columns available in dataset!")

    # **Check if required columns exist**
    required_columns = ["competency_id", "percentile_rank", "region", "incomegroup"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing Required Columns: {missing_cols}")

    # 🔹 Fix `competency_id` column (replace NaN with median)
    if "competency_id" in df.columns:
        df["competency_id"] = pd.to_numeric(df["competency_id"], errors="coerce")
        df["competency_id"].fillna(df["competency_id"].median(), inplace=True)

# 📌 **Load Trained Model**
try:
    model = joblib.load("ai_skill_rank_model.pkl")
    st.write("✅ Model Loaded Successfully!")
except FileNotFoundError:
    st.error("⚠ Model file not found. Please upload `ai_skill_rank_model.pkl` to GitHub.")
    model = None

# 📊 **Dataset Overview & Visualizations**
if df is not None and not df.empty:
    st.subheader("📊 AI Skill Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Bar Graph", "🔥 Heatmap", "📦 Boxplot"])

    # 📊 **Bar Graph**
    with tab1:
        st.subheader("📊 AI Skill Distribution by Region")
        region_options = df["region"].dropna().unique().tolist()
        selected_region = st.selectbox("🌍 Select a Region", region_options if region_options else ["No Data"], key="region_bar")

        filtered_df = df[df["region"] == selected_region]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=filtered_df["competency_id"], y=filtered_df["percentile_rank"], palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Competency ID")
        plt.ylabel("AI Skill Percentile Rank")
        plt.title(f"AI Skill Distribution in {selected_region}")
        st.pyplot(fig)

    # 🔥 **Heatmap - Correlation Matrix**
    with tab2:
        st.subheader("🔥 Correlation Heatmap of AI Skills")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠ No numerical columns available for correlation analysis.")

    # 📦 **Boxplot - AI Skill Percentile by Region**
    with tab3:
        st.subheader("📦 AI Skill Percentile Distribution by Region")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="region", y="percentile_rank", data=df, palette="coolwarm", ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Region")
        plt.ylabel("AI Skill Percentile Rank")
        plt.title("AI Skill Distribution by Region")
        st.pyplot(fig)

# 🎯 **AI Skill Rank Prediction**
st.subheader("🎯 AI Skill Rank Prediction")

region_options = df["region"].dropna().unique().tolist() if df is not None else ["No Data"]
income_options = df["incomegroup"].dropna().unique().tolist() if df is not None else ["No Data"]

region = st.selectbox("🌍 Select Region", region_options, key="region_select")
income_group = st.selectbox("💰 Select Income Group", income_options, key="income_select")
competency_id = st.slider("📈 Competency ID (Skill Level)", int(df["competency_id"].min()), int(df["competency_id"].max()), 5, key="competency_slider")

# ✅ **Feature Encoding (One-Hot Encoding)**
feature_vector = np.zeros(14)  

region_mapping = {region: i+6 for i, region in enumerate(df["region"].unique())}
if region in region_mapping:
    feature_vector[region_mapping[region]] = 1  

income_mapping = {income: i+12 for i, income in enumerate(df["incomegroup"].unique())}
if income_group in income_mapping:
    feature_vector[income_mapping[income_group]] = 1  

feature_vector[3] = competency_id  

user_input = np.array([feature_vector])

# 🎯 **Prediction**
st.subheader("📌 Model Prediction")

if st.button("Predict AI Skill Rank", key="predict_button_final"):
    if model is not None:
        try:
            prediction = model.predict(user_input)[0]
            avg_rank = df["percentile_rank"].mean() if df is not None else 0.5

            if prediction > avg_rank:
                st.success(f"🎯 **Predicted AI Skill Rank: {prediction:.2f}** 🚀 (Above Average!)")
            else:
                st.warning(f"⚠ **Predicted AI Skill Rank: {prediction:.2f}** 📉 (Below Average)")
        except Exception as e:
            st.error(f"⚠ Prediction failed: {e}")
    else:
        st.warning("⚠ Model is not loaded. Please check `ai_skill_rank_model.pkl`.")

# ✅ **Debugging Button**
if st.button("Run Code", key="run_code_button_final"):
    st.write("✅ **The code ran successfully!**")
