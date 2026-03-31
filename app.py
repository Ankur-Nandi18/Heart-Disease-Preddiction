import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Title
st.title("❤️ Heart Disease Risk Checker")
st.write("Enter the patient details below.")

st.info("⚠️ This tool is for educational purposes only and not a medical diagnosis.")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Basic Information")
    age = st.number_input("Age", 1, 120, 45)

    sex_option = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if sex_option == "Male" else 0

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Mild Chest Pain",
            "Moderate Chest Pain",
            "Non-Heart Related Pain",
            "No Chest Pain but Discomfort"
        ]
    )
    cp_dict = {
        "Mild Chest Pain": 0,
        "Moderate Chest Pain": 1,
        "Non-Heart Related Pain": 2,
        "No Chest Pain but Discomfort": 3
    }
    cp = cp_dict[cp_option]

    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 120)

    chol = st.number_input("Cholesterol Level (mg/dL)", 100, 600, 200)

with col2:
    st.markdown("### 🧪 Medical Details")

    fbs_value = st.number_input("Fasting Blood Sugar (mg/dL)", 70, 300, 100)
    fbs = 1 if fbs_value > 120 else 0

    restecg_option = st.selectbox(
        "Heart Electrical Test Result",
        ["Normal", "Minor Abnormality", "Serious Abnormality"]
    )
    restecg_dict = {
        "Normal": 0,
        "Minor Abnormality": 1,
        "Serious Abnormality": 2
    }
    restecg = restecg_dict[restecg_option]

    thalach = st.number_input("Maximum Heart Rate During Activity", 60, 220, 150)

    exang_option = st.selectbox(
        "Chest Pain During Exercise?",
        ["No", "Yes"]
    )
    exang = 1 if exang_option == "Yes" else 0

    oldpeak = st.number_input(
        "Heart Stress Level During Exercise (0 = Normal)",
        0.0, 10.0, 1.0
    )

    slope_option = st.selectbox(
        "Heart Recovery After Exercise",
        ["Normal Recovery", "Moderate Recovery", "Slow Recovery"]
    )
    slope_dict = {
        "Normal Recovery": 0,
        "Moderate Recovery": 1,
        "Slow Recovery": 2
    }
    slope = slope_dict[slope_option]

    ca = st.selectbox(
        "Number of Major Blood Vessels with Blockage",
        [0, 1, 2, 3, 4]
    )

    thal_option = st.selectbox(
        "Blood Flow Condition",
        [
            "Normal Blood Flow",
            "Fixed Problem in Blood Flow",
            "Reversible Blood Flow Issue"
        ]
    )
    thal_dict = {
        "Normal Blood Flow": 1,
        "Fixed Problem in Blood Flow": 2,
        "Reversible Blood Flow Issue": 3
    }
    thal = thal_dict[thal_option]

# Simple validation warnings
if chol > 400:
    st.warning("⚠️ Cholesterol level seems unusually high. Please verify.")

if trestbps > 180:
    st.warning("⚠️ Blood pressure is very high. Please verify.")

# Prediction
if st.button("🔍 Check Heart Risk"):
    features = np.array([[age, sex, cp, trestbps, chol,
                          fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    st.markdown("## 🩺 Result")

    # Probability (if supported)
    try:
        prob = model.predict_proba(scaled_features)[0][1]
        st.write(f"📊 Risk Probability: **{prob:.2f}**")
    except:
        pass

    if prediction[0] == 1:
        st.error("⚠️ The patient shows a HIGH RISK of heart disease. Please consult a doctor.")
    else:
        st.success("✅ The patient shows a LOW RISK of heart disease.")