import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# --- LOAD MODEL ---
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- SIDEBAR ---
st.sidebar.image("https://i.postimg.cc/tJBWhFpx/Top-Tech-Logo.png", width=200)
st.sidebar.markdown("## Bank Churn Prediction")
st.sidebar.write("This app predicts the likelihood of a customer churning.")

# --- MAIN TITLE ---
st.title("🏦 Customer Churn Prediction App")

st.markdown(
    """
    Use the form below to enter customer data and predict if they are likely to churn.
    """)

# --- USER INPUTS ---
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10)
balance = st.number_input("Account Balance", min_value=0.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
geography = st.selectbox("Geography", ['Germany', 'Spain', 'Other'])

# --- ENCODING ---
geo_germany = 1 if geography == 'Germany' else 0
geo_spain = 1 if geography == 'Spain' else 0
gender_binary = 1 if gender == 'Male' else 0

# --- DATAFRAME ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_binary],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography_Germany': [geo_germany],
    'Geography_Spain': [geo_spain],
})

# --- PREDICTION ---
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    result = "🟥 Exited" if prediction == 1 else "🟩 Not Exited"

    st.subheader("🔎 Result")
    st.write(f"**Prediction:** {result}")
    st.write(f"**Churn Probability:** {probability:.2%}")
