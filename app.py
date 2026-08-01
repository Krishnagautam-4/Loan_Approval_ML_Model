import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="AI Loan Approval Portal",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Smart Loan Approval Portal")
st.caption("Powered by Logistic Regression Machine Learning Model")
st.markdown("---")

# 2. Load Model and Features safely
@st.cache_resource
def load_model_data():
    saved_data = joblib.load('loan_model.pkl')
    return saved_data['model'], saved_data['feature_names']

try:
    model, feature_names = load_model_data()
except Exception as e:
    st.error("Model file not found or corrupted. Please run Cell 10 first.")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("📋 Applicant Profile")

input_mode = st.sidebar.radio("Input Method:", ["Interactive Sliders", "Manual Typing"])

if input_mode == "Interactive Sliders":
    age = st.sidebar.slider("Age", 18, 100, 25)
    income = st.sidebar.slider("Annual Income ($)", 5000, 200000, 50000, step=1000)
    emp_exp = st.sidebar.slider("Work Experience (Years)", 0, 50, 3)
    loan_amount = st.sidebar.slider("Requested Loan Amount ($)", 500, 50000, 10000, step=500)
    int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 11.0, step=0.1)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    cred_hist = st.sidebar.slider("Credit History Length (Years)", 0, 30, 3)
else:
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
    income = st.sidebar.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=50000, step=1000)
    emp_exp = st.sidebar.number_input("Work Experience (Years)", min_value=0, max_value=50, value=3)
    loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=10000, step=500)
    int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.0, step=0.1)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=650)
    cred_hist = st.sidebar.number_input("Credit History Length (Years)", min_value=0, max_value=30, value=3)

# Categorical Dropdowns
st.sidebar.subheader("Personal Details & Intent")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", ["Bachelor", "Doctorate", "High School", "Master", "Associate/Other"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "OTHER", "MORTGAGE"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBT CONSOLIDATION"])
previous_defaults = st.sidebar.selectbox("Previous Loan Defaults?", ["No", "Yes"])

loan_percent_income = round(loan_amount / income if income > 0 else 0, 2)

# Dashboard Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Income", f"${income:,}")
col2.metric("Requested Loan", f"${loan_amount:,}")
col3.metric("Loan/Income Ratio", f"{loan_percent_income * 100:.1f}%")
col4.metric("Credit Score", credit_score)

st.markdown("---")

# 4. Construct feature mapping to guarantee exact match with dataset columns
input_dict = {
    'person_age': float(age),
    'person_income': float(income),
    'person_emp_exp': int(emp_exp),
    'loan_amnt': float(loan_amount),
    'loan_int_rate': float(int_rate),
    'loan_percent_income': float(loan_percent_income),
    'cb_person_cred_hist_length': float(cred_hist),
    'credit_score': int(credit_score),
    'person_gender_male': 1 if gender == "Male" else 0,
    'person_education_Bachelor': 1 if education == "Bachelor" else 0,
    'person_education_Doctorate': 1 if education == "Doctorate" else 0,
    'person_education_High School': 1 if education == "High School" else 0,
    'person_education_Master': 1 if education == "Master" else 0,
    'person_home_ownership_OTHER': 1 if home_ownership == "OTHER" else 0,
    'person_home_ownership_OWN': 1 if home_ownership == "OWN" else 0,
    'person_home_ownership_RENT': 1 if home_ownership == "RENT" else 0,
    'loan_intent_EDUCATION': 1 if loan_intent == "EDUCATION" else 0,
    'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
    'loan_intent_MEDICAL': 1 if loan_intent == "MEDICAL" else 0,
    'loan_intent_PERSONAL': 1 if loan_intent == "PERSONAL" else 0,
    'loan_intent_VENTURE': 1 if loan_intent == "VENTURE" else 0,
    'previous_loan_defaults_on_file_Yes': 1 if previous_defaults == "Yes" else 0
}

# Create DataFrame and reorder columns to match training features exactly
input_df = pd.DataFrame([input_dict])[feature_names]

# 5. Prediction Execution
st.subheader("⚡ Automated Underwriting Decision")

if st.button("Evaluate Loan Application"):
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        approval_prob = probabilities[1] * 100

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.success("### 🎉 Result: LOAN APPROVED")
                st.write(f"Approval Probability: **{approval_prob:.1f}%**")
            else:
                st.error("### ❌ Result: LOAN REJECTED")
                st.write(f"Approval Probability: **{approval_prob:.1f}%**")

        with res_col2:
            st.write("Confidence Level")
            st.progress(int(approval_prob))
    except Exception as err:
        st.error(f"Prediction Error: {err}")
