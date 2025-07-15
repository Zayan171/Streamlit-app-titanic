import streamlit as st
import pickle
import numpy as np

# 🚨 Correct file name for your model
model = pickle.load(open('save_titanic_model.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction App")

st.write("Fill out the details below to predict if the passenger will survive or not.")

# 🧑 Passenger features
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encoding
sex = 1 if sex == "male" else 0
embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_dict[embarked]

# Prepare features
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ Passenger will SURVIVE! 🎉")
    else:
        st.error("💔 Passenger will NOT survive.")

