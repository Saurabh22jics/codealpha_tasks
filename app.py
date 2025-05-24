# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger information:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.slider("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", min_value=0.0, value=32.0)

if st.button("Predict"):
    sex_val = 0 if sex == "male" else 1
    input_df = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare]],
                            columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    
    prediction = model.predict(input_df)[0]
    result = "Survived ✅" if prediction == 1 else "Did Not Survive ❌"
    st.subheader(f"Prediction: {result}")
