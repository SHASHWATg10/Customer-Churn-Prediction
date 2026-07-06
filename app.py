import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color:#f8f9fa;
}

h1{
    color:#0f62fe;
    text-align:center;
}

.stButton>button{
    background:#0f62fe;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}

.stDownloadButton>button{
    background:green;
    color:white;
}

.result{
    padding:20px;
    border-radius:12px;
    font-size:22px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model.\n\n{e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Customer Churn")

st.sidebar.markdown("""
### About

This application predicts whether a customer is likely to churn using a trained Machine Learning Decision Tree Pipeline.

### Model
Decision Tree Pipeline

### Features
- Probability Prediction
- Download Results
- Professional UI
""")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction")

st.write("Fill customer information below and click **Predict**.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0,1]
        )

        Partner = st.selectbox(
            "Partner",
            ["Yes","No"]
        )

        Dependents = st.selectbox(
            "Dependents",
            ["Yes","No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            0,
            100,
            12
        )

        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes","No"]
        )

        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["Yes","No","No phone service"]
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL","Fiber optic","No"]
        )

        OnlineSecurity = st.selectbox(
            "Online Security",
            ["Yes","No","No internet service"]
        )

        OnlineBackup = st.selectbox(
            "Online Backup",
            ["Yes","No","No internet service"]
        )

    with col2:

        DeviceProtection = st.selectbox(
            "Device Protection",
            ["Yes","No","No internet service"]
        )

        TechSupport = st.selectbox(
            "Tech Support",
            ["Yes","No","No internet service"]
        )

        StreamingTV = st.selectbox(
            "Streaming TV",
            ["Yes","No","No internet service"]
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["Yes","No","No internet service"]
        )

        Contract = st.selectbox(
            "Contract",
            ["Month-to-month","One year","Two year"]
        )

        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes","No"]
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        MonthlyCharges = st.number_input(
            "Monthly Charges",
            0.0,
            1000.0,
            70.0
        )

        TotalCharges = st.number_input(
            "Total Charges",
            0.0,
            100000.0,
            1000.0
        )

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    try:

        input_df = pd.DataFrame({
            "gender":[gender],
            "SeniorCitizen":[SeniorCitizen],
            "Partner":[Partner],
            "Dependents":[Dependents],
            "tenure":[tenure],
            "PhoneService":[PhoneService],
            "MultipleLines":[MultipleLines],
            "InternetService":[InternetService],
            "OnlineSecurity":[OnlineSecurity],
            "OnlineBackup":[OnlineBackup],
            "DeviceProtection":[DeviceProtection],
            "TechSupport":[TechSupport],
            "StreamingTV":[StreamingTV],
            "StreamingMovies":[StreamingMovies],
            "Contract":[Contract],
            "PaperlessBilling":[PaperlessBilling],
            "PaymentMethod":[PaymentMethod],
            "MonthlyCharges":[MonthlyCharges],
            "TotalCharges":[TotalCharges]
        })

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        churn_probability = probability[1] * 100
        stay_probability = probability[0] * 100

        st.markdown("---")

        if prediction == 1:
            st.error(f"⚠ Customer is likely to CHURN")
        else:
            st.success(f"✅ Customer is likely to STAY")

        col1,col2 = st.columns(2)

        with col1:
            st.metric(
                "Stay Probability",
                f"{stay_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%"
            )

        result = input_df.copy()
        result["Prediction"] = prediction
        result["Stay Probability"] = stay_probability
        result["Churn Probability"] = churn_probability
        result["Timestamp"] = datetime.now()

        csv = result.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download Prediction",
            csv,
            file_name="customer_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)