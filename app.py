import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("./train.csv")

st.header("AUTISM Prediction")

st.subheader("Enter the following fields")

age = st.number_input("Enter the Age :", value=0, step=1)
gender = st.selectbox("Gender 1 for Male and 0 for Female", [0, 1])
enthicity = st.number_input("Ethnicity :", value=0, step=1)
jaundice = st.selectbox("Jaundice 1 for Yes and 0 for No", [0, 1])
country = st.number_input("Country :", value=0, step=1)
realtion = st.number_input("Relation :", value=0, step=1)
agegroup = st.number_input("Age Group :", value=0, step=1)
result = st.number_input("Result :", value=0, step=1)
sum_score = st.number_input("Enter the Sum Score :", value=0, step=1)
ind = st.number_input("Enter the Index :", value=0, step=1)

st.subheader("Autism Screening Scores (0 or 1)")
a1_score = st.number_input("Enter the A1 score :", value=0, step=1)
a2_score = st.number_input("Enter the A2 score :", value=0, step=1)
a3_score = st.number_input("Enter the A3 score :", value=0, step=1)
a4_score = st.number_input("Enter the A4 score :", value=0, step=1)
a5_score = st.number_input("Enter the A5 score :", value=0, step=1)
a6_score = st.number_input("Enter the A6 score :", value=0, step=1)
a7_score = st.number_input("Enter the A7 score :", value=0, step=1)
a8_score = st.number_input("Enter the A8 score :", value=0, step=1)
a9_score = st.number_input("Enter the A9 score :", value=0, step=1)
a10_score = st.number_input("Enter the A10 score :", value=0, step=1)

def asd():
    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)
    with open("asd_model.pkl", "rb") as f:
        model = joblib.load(f)
    scores = [a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score, age, gender, enthicity, jaundice, country, result, realtion, agegroup, sum_score, ind]
    scores = np.array(scores).reshape(1, -1)
    scores = scaler.transform(scores)
    final_result = model.predict(scores)[0]
    return final_result

def test():
    array = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 15, 0, 3, 0, 1, 0, 5, 2, 0, 0]
    array = np.array(array).reshape(1, -1)
    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)
    with open("asd_model.pkl", "rb") as f:
        model = joblib.load(f)
    array = scaler.transform(array)
    final_result = model.predict(array)[0]
    return final_result

if st.button(label="SUBMIT"):
    final_result = asd()
    st.subheader("ASD Prediction : YES" if final_result == 1 else "ASD Prediction : NO")

# if st.button(label="TEST"):
#     final_result = test()
#     st.subheader("ASD Prediction : YES" if final_result == 1 else "ASD Prediction : NO")