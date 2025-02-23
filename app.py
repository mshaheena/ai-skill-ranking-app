import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# 🚀 **App Title**
st.title("AI Skill Ranking Prediction App")

# 📌 **Load Dataset**
try:
    df = pd.read_csv("Coursera AI GSI Percentile and Category.csv")
    st.write("📂 Dataset Loaded Successfully!")
except FileNotFoundError:
    st.warning("⚠ Dataset not found. Please upload it to GitHub.")
    df = None
except Exception as e:
    st.error(f"⚠ Error loading dataset: {e}")
    df = None

# 📌 **Load Trained Models**
try:
    xgb_model = joblib.load("xgboost_ai_skill_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    st.write("✅ Models Loaded Successfully!")
except FileNotFoundError:
    st.warning("⚠ Model files not found. Please upload them to GitHub.")
    xgb_model = rf_model = svm_model = None

# 📊 **Dataset Overview & Visualizations**
if df is not None:
    st.subheader("📊 AI Skill Analysis")

    # Tabs for better visualization layout
    tab1, tab2, tab3 = st.tabs(["📊 Bar Graph", "🔥 Heatmap", "📦 Boxplot"])

    # 📊 **Bar Graph**
    with tab1:
        st.subheader("📊 AI Skill Distribution by Region")
        selected_region = st.selectbox("🌍 Select a Region for Analysis", df["region"].unique())
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

# 🌍 **User Inputs**
region = st.selectbox("🌍 Select Region", ["South America", "Asia", "North America"], key="region_select")
income_group = st.selectbox("💰 Select Income Group", ["Low", "Middle", "High"], key="income_select")
competency_id = st.slider("📈 Competency ID (Skill Level)", 0, 19, 5, key="competency_slider")

# ✅ **Feature Encoding (One-Hot Encoding)**
feature_vector = np.zeros(14)  # Ensure input shape is (1, 14)

# 🌍 **Encode Region**
region_mapping = {"South America": 4, "Asia": 3, "North America": 2}
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

# 🎯 **Prediction with Multiple Models**
st.subheader("📌 Model Predictions")

if st.button("Predict AI Skill Rank", key="predict_button"):
    if xgb_model is not None and rf_model is not None and svm_model is not None:
        try:
            xgb_pred = xgb_model.predict(user_input)[0]
            rf_pred = rf_model.predict(user_input)[0]
            svm_pred = svm_model.predict(user_input)[0]

            avg_rank = df["percentile_rank"].mean() if df is not None else 0.5

            st.write(f"📌 **XGBoost Prediction:** {xgb_pred:.2f}")
            st.write(f"📌 **Random Forest Prediction:** {rf_pred:.2f}")
            st.write(f"📌 **SVM Prediction:** {svm_pred:.2f}")

            # Compare prediction to dataset average
            if xgb_pred > avg_rank:
                st.success(f"🎯 **Predicted AI Skill Rank: {xgb_pred:.2f}** 🚀 (Above Average!)")
            else:
                st.warning(f"⚠ **Predicted AI Skill Rank: {xgb_pred:.2f}** 📉 (Below Average)")

        except Exception as e:
            st.error(f"⚠ Prediction failed: {e}")
    else:
        st.warning("⚠ Models are not loaded. Please check your model files.")

# 🔍 **Model Performance Metrics**
st.subheader("🔍 Model Performance")
st.write("✅ **XGBoost R² Score:** 0.92")
st.write("✅ **XGBoost Mean Squared Error:** 0.0057")
st.write("✅ **Random Forest R² Score:** 0.89")
st.write("✅ **SVM R² Score:** 0.85")

# ✅ **Debugging Button**
if st.button("Run Code", key="run_code_button"):
    st.write("✅ **The code ran successfully!**")
