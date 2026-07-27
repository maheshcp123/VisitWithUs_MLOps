import streamlit as st
import pandas as pd
import pickle

st.title("Visit with Us: Wellness Package Predictor")

# Load model
model = pickle.load(open('models/best_model.pkl', 'rb'))

# Inputs
age = st.slider("Age", 18, 70, 30)
income = st.number_input("Monthly Income", value=20000)
passport = st.selectbox("Has Passport?", [0, 1])

# Create DF for prediction
input_data = pd.DataFrame([[age, income, passport]], columns=['Age', 'MonthlyIncome', 'Passport'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Likely to Purchase" if prediction[0] == 1 else "Not Likely"
    st.success(f"Prediction: {result}")
