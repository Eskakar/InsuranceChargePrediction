import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# LOAD MODEL & TRANSFORMERS
# ================================
model = joblib.load("model_insurance.pkl")
scaler = joblib.load("scaler.pkl")
labelEncod_sex = joblib.load("labelencoder_sex.pkl")
model_features = joblib.load("model_features.pkl")       # urutan fitur final
ohe_columns = joblib.load("ohe_columns.pkl")             # kolom hasil OHE
numerical_cols = joblib.load("numerical_cols.pkl")       # age, bmi, children

# ================================
# STREAMLIT UI
# ================================
st.title("Prediksi Medical Insurance Charges 💵")
st.write("Masukkan data berikut untuk memprediksi biaya asuransi (charges).")

# ----- INPUT USER -----
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ================================
# PREDICTION
# ================================
if st.button("Prediksi Charges"):
    # Buat dataframe dari input user
    new_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # -----------------------
    # LABEL ENCODE (sex)
    # -----------------------
    new_data["sex"] = labelEncod_sex.transform(new_data["sex"])

    # -----------------------
    # ONE HOT ENCODING (smoker, region)
    # -----------------------
    new_data = pd.get_dummies(new_data, columns=["smoker", "region"])

    # Pastikan kolom OHE sesuai model
    for col in ohe_columns:
        if col not in new_data:
            new_data[col] = 0

    # -----------------------
    # SCALING fitur numerik
    # -----------------------
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

    # -----------------------
    # SUSUN URUTAN FITUR
    # -----------------------
    new_data = new_data.reindex(columns=model_features, fill_value=0)

    # -----------------------
    # PREDIKSI
    # -----------------------
    prediction = model.predict(new_data)[0]

    st.success(f"Prediksi Medical Charges: **${prediction:,.2f}**")
