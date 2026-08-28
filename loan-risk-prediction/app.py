import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/loan_pipeline.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="🏦"
)

st.title("🏦 Loan Risk Predictor")

st.write(
    "Enter the applicant details below to predict loan approval."
)


# ============================================================
# APPLICANT DETAILS
# ============================================================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)

loan_term = st.number_input(
    "Loan Term (months)",
    min_value=0,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Rural", "Semiurban"]
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Loan Status"):

    new_applicant = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })


    # Make prediction
    prediction = model.predict(new_applicant)


    # Get probabilities
    probabilities = model.predict_proba(new_applicant)

    classes = model.classes_


    # Display result
    if prediction[0] == "Y":

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Not Approved")


    # Display probability
    st.subheader("Prediction Probability")

    for class_name, probability in zip(classes, probabilities[0]):

        if class_name == "Y":
            label = "Approved"

        else:
            label = "Not Approved"

        st.write(
            f"{label}: {probability * 100:.2f}%"
        )