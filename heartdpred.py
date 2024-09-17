import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict heart disease
def predict_heart_disease(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit app
st.title('Heart Disease Prediction App')

# Input features
age = st.number_input('Age', min_value=0, max_value=120, value=25)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])  # Assuming numeric encoding
trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholestoral', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

# Prepare input data
features = np.array([age, sex == 'Male', cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

# Make prediction
if st.button('Predict'):
    result = predict_heart_disease(features)
    st.write('Prediction: ', 'Heart Disease' if result == 1 else 'No Heart Disease')
