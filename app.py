import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and encoders
model = joblib.load("employee_attrition.pkl")
label_encoder = joblib.load("label_encoder.pkl")   # dictionary of encoders
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit UI
st.title("Employee Attrition Prediction")
st.markdown("Enter the employee details to predict whether they are likely to leave the company.")

st.sidebar.header("Employee Details")

# Input function
def user_input_features():
    inputs = {}

    inputs['Age'] = st.sidebar.number_input("Age", 18, 65, 30)
    inputs['MonthlyIncome'] = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
    inputs['JobSatisfaction'] = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
    inputs['OverTime'] = st.sidebar.selectbox("Over Time", ["Yes", "No"])
    inputs['DistanceFromHome'] = st.sidebar.number_input("Distance From Home", 0, 50, 10)

    # Match training features
    data = {}
    for col in feature_columns:
        if col in inputs:
            data[col] = inputs[col]
        else:
            data[col] = 0  # default for missing features

    return pd.DataFrame(data, index=[0])

# Get input
input_df = user_input_features()

# 🔥 FIXED ENCODING (MAIN FIX)
for col, encoder in label_encoder.items():
    if col in input_df.columns:
        try:
            input_df[col] = label_encoder.transform(input_df[col].astype(str))
        except:
            input_df[col] = 0

# Ensure all data is numeric
input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Prediction
if st.button("Predict Attrition"):
    
    # Debug (optional)
    # st.write(input_df)
    # st.write(input_df.dtypes)

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")

    if prediction[0] == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with the company.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of leaving: {prediction_proba[0][1]:.2f}")
