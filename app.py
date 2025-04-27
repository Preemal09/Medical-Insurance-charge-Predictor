import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Model
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Preprocess Mappings
region_map = {'Northwest': 0, 'Northeast': 1, 'Southeast': 2, 'Southwest': 3}
sex_map = {'Male': 0, 'Female': 1}
smoker_map = {'Yes': 1, 'No': 0}

# Streamlit UI
st.title("Insurance Charge Prediction")

# Input Fields
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

# Predict Button
if st.button("Predict Charges"):
    try:
        # Create the input dataframe
        new_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # Preprocess inputs
        new_data['smoker'] = new_data['smoker'].map(smoker_map)
        
        # Only keep the columns used for training
        final_data = new_data[['age', 'bmi', 'children', 'smoker']]

        # Predict the charges
        prediction = model.predict(final_data)[0]
        st.success(f"Predicted Charges: ${round(prediction, 2)}")
    except Exception as e:
        st.error(f"Error predicting charges: {e}")
