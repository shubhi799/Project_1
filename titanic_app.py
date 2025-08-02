import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

# Convert inputs to numbers
sex = 0 if sex == "Male" else 1
embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

# Create input DataFrame
data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# Predict
if st.button("Predict"):
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("❌ Unfortunately, this passenger would NOT have survived.")
