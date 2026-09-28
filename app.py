#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Diagnosis Prediction",
    page_icon="🩺",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 15px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.footer {
    text-align: center;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🩺 Diabetes Diagnosis Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Diabetes Prediction System</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Patient Information
# -----------------------------
st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.write("Enter the patient's medical information below.")

col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    # -----------------------------
    # Result
    # -----------------------------

    if prediction == 1:

        st.error("⚠️ Diabetes Predicted")

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <p>Diabetes is predicted based on the entered information.</p>
                <h3>Probability: {probability:.2%}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.success("✅ No Diabetes Predicted")

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <p>No diabetes is predicted based on the entered information.</p>
                <h3>Probability: {probability:.2%}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Project Information
# -----------------------------
st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("""
    This project uses a Machine Learning classification model
    to predict diabetes based on patient medical information.

    **Model Used:** Logistic Regression

    **Input Features:**
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age

    **Target:** Diabetes Outcome
    """)


# -----------------------------
# Disclaimer
# -----------------------------
st.markdown(
    """
    <div class="footer">
    ⚠️ This application is developed for educational and demonstration
    purposes only. It should not be used as a substitute for professional
    medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

