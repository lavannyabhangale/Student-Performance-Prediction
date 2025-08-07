import streamlit as st
import pandas as pd
import joblib

# Load the trained model & label encoders dictionary
model = joblib.load("models/student_performance_model.pkl")
label_encoders = joblib.load("models/label_encoder.pkl")  # dictionary of LabelEncoders

st.title("🎓 Student Performance Prediction")

# Input fields (removed Math Score)
gender = st.selectbox("Gender", ["female", "male"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_edu = st.selectbox("Parental Education Level", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation", ["none", "completed"])
reading_score = st.number_input("Reading Score", 0, 100)
writing_score = st.number_input("Writing Score", 0, 100)

if st.button("Predict Performance"):
    # Prepare input data (no Math Score here)
    input_data = pd.DataFrame([[
        gender, race_ethnicity, parental_edu, lunch, test_prep,
        reading_score, writing_score
    ]], columns=[
        "gender", "race/ethnicity", "parental level of education",
        "lunch", "test preparation course",
        "reading score", "writing score"
    ])

    # Encode categorical columns using saved encoders
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Predict category (0, 1, 2)
    pred_class = model.predict(input_data)[0]

    # Decode prediction back to original label (Low, Medium, High)
    performance_labels = {0: "Low", 1: "Medium", 2: "High"}
    prediction_label = performance_labels.get(pred_class, "Unknown")

    st.success(f"Predicted Performance: {prediction_label}")
