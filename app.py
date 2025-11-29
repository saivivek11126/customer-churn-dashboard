import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction")

# Load dataset for business metrics
import pandas as pd
df = pd.read_csv("teleco-dataset.csv")

st.subheader("Customer Churn Distribution")
churn_counts = df['Churn'].value_counts()
st.bar_chart(churn_counts)

st.subheader("Monthly Charges Distribution")
st.bar_chart(df['MonthlyCharges'])

st.subheader("Tenure Distribution")
st.bar_chart(df['tenure'])

total_customers = df.shape[0]
churned_customers = df[df['Churn']==1].shape[0]
churn_percent = round((churned_customers / total_customers)*100, 2)
avg_tenure = round(df['tenure'].mean(), 2)
avg_monthly = round(df['MonthlyCharges'].mean(), 2)

st.subheader("Business Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Churn %", f"{churn_percent}%")
col4.metric("Average Tenure", avg_tenure)
col5.metric("Average Monthly Charges", avg_monthly)

st.write("Enter customer details below:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# Encode inputs same way as training
def encode_input():
    mapping = {
        "Yes":1,
        "No":0,
        "Male":1,
        "Female":0,
        "DSL":0,
        "Fiber optic":1,
        "No":2,
        "Month-to-month":0,
        "One year":1,
        "Two year":2,
        "Electronic check":0,
        "Mailed check":1,
        "Bank transfer (automatic)":2,
        "Credit card (automatic)":3,
        "No phone service":2,
        "No internet service":2
    }
    input_list = [
        mapping[gender],
        SeniorCitizen,
        mapping[Partner],
        mapping[Dependents],
        tenure,
        mapping[PhoneService],
        mapping[MultipleLines],
        mapping[InternetService],
        mapping[OnlineSecurity],
        mapping[OnlineBackup],
        mapping[DeviceProtection],
        mapping[TechSupport],
        mapping[StreamingTV],
        mapping[StreamingMovies],
        mapping[Contract],
        mapping[PaperlessBilling],
        mapping[PaymentMethod],
        MonthlyCharges,
        TotalCharges
    ]
    return np.array(input_list).reshape(1,-1)

if st.button("Predict Churn"):
    input_data = encode_input()
    prediction = model.predict(input_data)
    if prediction[0]==1:
        st.error("Customer is likely to **churn** 😞")
    else:
        st.success("Customer is **not likely to churn** 🙂")

