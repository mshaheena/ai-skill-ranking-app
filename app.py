

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


