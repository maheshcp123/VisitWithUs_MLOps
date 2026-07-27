import streamlit as st
import pandas as pd
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Visit with Us - Wellness Predictor", layout="wide")

# Title and Context
st.title("💼 Wellness Tourism Package Prediction")
st.markdown("""
This app predicts the likelihood of a customer purchasing the new **Wellness Tourism Package** 
based on their profile and interaction history.
""")

# Load the trained model
MODEL_PATH = 'models/best_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("Model file not found! Please ensure the MLOps pipeline has run and committed the model.")
else:
    # Sidebar for Inputs
    st.sidebar.header("Customer Details")
    
    # Numeric Inputs
    age = st.sidebar.slider("Age", 18, 70, 35)
    city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    person_visiting = st.sidebar.number_input("Number of Persons Visiting", 1, 10, 2)
    star_rating = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
    trips = st.sidebar.number_input("Annual Number of Trips", 1, 20, 3)
    passport = st.sidebar.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])
    own_car = st.sidebar.selectbox("Owns a Car? (1=Yes, 0=No)", [0, 1])
    children = st.sidebar.number_input("Number of Children Visiting", 0, 5, 0)
    income = st.sidebar.number_input("Monthly Income (Rs)", value=25000)

    st.sidebar.header("Interaction Data")
    satisfaction = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
    followups = st.sidebar.number_input("Number of Follow-ups", 1, 10, 3)
    pitch_duration = st.sidebar.number_input("Duration of Pitch (min)", 5, 120, 15)

    # Categorical Inputs
    contact = st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Freelancer", "Large Business"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

    # Create Input Dataframe
    # Note: We must include all columns used during training
    input_dict = {
        'Age': age,
        'TypeofContact': contact,
        'CityTier': city_tier,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': person_visiting,
        'PreferredPropertyStar': star_rating,
        'MaritalStatus': marital,
        'NumberOfTrips': trips,
        'Passport': passport,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': children,
        'Designation': designation,
        'MonthlyIncome': income,
        'PitchSatisfactionScore': satisfaction,
        'NumberOfFollowups': followups,
        'DurationOfPitch': pitch_duration
    }
    
    input_df = pd.DataFrame([input_dict])

    # Data Preprocessing (Must match Training script)
    # 1. Handle Categorical Encoding
    input_encoded = pd.get_dummies(input_df)
    
    # 2. Align with Training Features
    # We retrieve the feature names the model was trained on
    model_features = model.feature_names_in_
    
    # Add missing columns with 0, and remove extra columns
    input_final = input_encoded.reindex(columns=model_features, fill_value=0)

    # Prediction
    st.subheader("Prediction Result")
    if st.button("Analyze Potential"):
        prediction = model.predict(input_final)
        probability = model.predict_proba(input_final)[:, 1]

        if prediction[0] == 1:
            st.success(f"🎯 **High Potential**: Customer is likely to purchase the Wellness Package.")
            st.metric("Purchase Probability", f"{round(probability[0]*100, 2)}%")
        else:
            st.warning(f"⚠️ **Low Potential**: Customer is unlikely to purchase at this time.")
            st.metric("Purchase Probability", f"{round(probability[0]*100, 2)}%")

    # Display Input Diagnostics
    with st.expander("View Input Dataframe (Cleaned)"):
        st.write(input_final)
