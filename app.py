import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk System", layout="centered")

# Header
st.title("🏥 Heart Disease Risk Assessment System")
st.write("Fill the details below for medical risk evaluation.")
st.markdown("---")

# Session state
if "data" not in st.session_state:
    st.session_state.data = {}
# Demo buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("💚 Fill Low Risk Data"):
        st.session_state.data = {
            "age": 30,
            "gender": "Female",
            "bp": 110,
            "chol": 180,
            "sugar": 90,
            "hr": 170,
            "cp": "Non-Heart Related",
            "ex": "No",
            "stress": 0.5,
            "rec": "Normal",
            "ves": 0,
            "thal": "Normal"
        }
with col2:
    if st.button("❤️ Fill High Risk Data"):
        st.session_state.data = {
            "age": 60,
            "gender": "Male",
            "bp": 160,
            "chol": 300,
            "sugar": 180,
            "hr": 120,
            "cp": "Severe",
            "ex": "Yes",
            "stress": 4.0,
            "rec": "Slow",
            "ves": 3,
            "thal": "Reversible Issue"
        }

data = st.session_state.data
# Inputs
age = st.number_input("Age", 1, 120, data.get("age", 45))

gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    index=0 if data.get("gender", "Male") == "Male" else 1
)
sex = 1 if gender == "Male" else 0

bp = st.number_input("Resting Blood Pressure", 80, 250, data.get("bp", 120))
chol = st.number_input("Cholesterol", 100, 600, data.get("chol", 200))

sugar_value = st.number_input("Fasting Blood Sugar", 70, 300, data.get("sugar", 100))
fbs = 1 if sugar_value > 120 else 0

heart_rate = st.number_input("Max Heart Rate", 60, 220, data.get("hr", 150))

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Mild", "Moderate", "Non-Heart Related", "Severe"],
    index=["Mild", "Moderate", "Non-Heart Related", "Severe"].index(data.get("cp", "Mild"))
)
cp_dict = {
    "Mild": 0,
    "Moderate": 1,
    "Non-Heart Related": 2,
    "Severe": 3
}
cp = cp_dict[chest_pain]

exercise_pain = st.selectbox(
    "Chest Pain During Exercise?",
    ["No", "Yes"],
    index=0 if data.get("ex", "No") == "No" else 1
)
exang = 1 if exercise_pain == "Yes" else 0

stress_level = st.slider("Heart Stress Level", 0.0, 5.0, data.get("stress", 1.0))
oldpeak = stress_level

recovery = st.selectbox(
    "Recovery",
    ["Normal", "Moderate", "Slow"],
    index=["Normal", "Moderate", "Slow"].index(data.get("rec", "Normal"))
)
slope_dict = {
    "Normal": 0,
    "Moderate": 1,
    "Slow": 2
}
slope = slope_dict[recovery]

vessels = st.selectbox("Blocked Vessels", [0, 1, 2, 3, 4], index=data.get("ves", 0))
ca = vessels

thal_option = st.selectbox(
    "Blood Flow",
    ["Normal", "Fixed Issue", "Reversible Issue"],
    index=["Normal", "Fixed Issue", "Reversible Issue"].index(data.get("thal", "Normal"))
)

thal_dict = {
    "Normal": 1,
    "Fixed Issue": 2,
    "Reversible Issue": 3
}
thal = thal_dict[thal_option]

restecg = 0
# Prediction
if st.button("🔍 Analyze Risk"):

    features = np.array([[age, sex, cp, bp, chol,
                          fbs, restecg, heart_rate,
                          exang, oldpeak, slope, ca, thal]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]

    low_risk_probability = probabilities[0] * 100
    high_risk_probability = probabilities[1] * 100

    st.markdown("---")
    st.subheader("🩺 Risk Assessment Result")

    st.progress(int(high_risk_probability))
    st.write(f"### ❤️ High Risk Probability: {high_risk_probability:.2f}%")
    st.write(f"### 💚 Low Risk Probability: {low_risk_probability:.2f}%")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")

        if high_risk_probability >= 80:
            st.warning("🚨 Very High Risk! Immediate doctor consultation recommended.")
        elif high_risk_probability >= 60:
            st.warning("⚠️ Moderate to High Risk. Medical advice is recommended.")

        st.markdown("""
            ### 🏥 Doctor Recommendation:
            - Consult a Cardiologist immediately
            - Maintain a low-fat healthy diet
            - Avoid smoking and alcohol
            - Monitor blood pressure regularly
            - Maintain sugar and cholesterol levels
            - Daily 30 minutes walking
            - ECG and blood tests if needed
            """)
    else:
        st.success("✅ Low Risk of Heart Disease")

        st.markdown("""
                ### 💚 Health Advice:
                - Continue regular exercise
                - Maintain balanced diet
                - Drink enough water
                - Sleep at least 7-8 hours
                - Regular yearly health check-up
                - Monitor blood pressure and sugar
                """)