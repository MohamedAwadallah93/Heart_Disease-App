import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Loading the trained model
with open('LogisticReg_heart.pkl', 'rb') as file:
    model = pickle.load(file)

# 2. App Title and Description
st.title("❤️ Heart Disease Prediction App")
st.subheader("By : Mohamed Awadallah , moh.alsafi321@gmail.com")

st.write("Enter the patient details below to predict the risk of Cardiovascular Disease (CVD).")

# 3. Input Fields

col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    

with col2:
    systolic = st.number_input("Systolic Blood Pressure", min_value=60, max_value=250, value=120)
    diastolic = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], help="1: Normal, 2: Above Normal, 3: Well Above Normal")
    gluc = st.selectbox("Glucose Level : 1/ Normal, 2/ High, 3/ Very high)", options=[1, 2, 3])

smoke = st.radio("Do you smoke?", options=["Yes", "No"])
alco = st.radio("Do you consume alcohol?", options=["Yes", "No"])
active = st.radio("Are you physically active?", options=["Yes", "No"])

# 4. Preprocess Inputs to match training format

def preprocess_input():
    
    gender_val = 0 if gender == "Male" else 1 
    smoke_val = 1 if smoke == "Yes" else 0
    alco_val = 1 if alco == "Yes" else 0
    active_val = 1 if active == "Yes" else 0
    
    # A list of features in the same order as X_train
    features = [age_years, gender_val, height, weight, systolic, diastolic, cholesterol, gluc, smoke_val, alco_val, active_val]
    return np.array([features])

# 5. Prediction Button
if st.button("Predict"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Results:")
    if prediction[0] == 1 or prediction[0] == "Yes":
        st.error(f"High Risk: The model predicts a high probability of Heart Disease.")
    else:
        st.success(f"Low Risk: The model predicts a low probability of Heart Disease.")
        
    st.write(f"Confidence Level: {np.max(prediction_proba)*100:.2f}%")
