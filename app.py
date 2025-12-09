import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD MODEL & ARTIFACTS
# =========================
rf_model = joblib.load("model_insurance.pkl")
scaler = joblib.load("scaler.pkl")
ohe_columns = joblib.load("ohe_columns.pkl")
numerical_cols = joblib.load("numerical_cols.pkl")
model_features = joblib.load("model_features.pkl")   # memastikan urutan fitur konsisten

# =========================
# APLIKASI STREAMLIT
# =========================
st.title("🔵 Analisis Prediksi Biaya Medis Personal")
st.write("Model prediksi menggunakan Random Forest pada data asuransi kesehatan.")

st.markdown("""
Aplikasi ini memprediksi **biaya medis personal (charges)** berdasarkan faktor:
- Usia  
- BMI  
- Jumlah anak  
- Status perokok  
- Wilayah tempat tinggal  
""")

st.header("✨ Masukkan Data Pengguna")


# =========================
# INPUT FORM
# =========================
age = st.number_input("Usia", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0)

smoker = st.selectbox("Apakah Perokok?", ["Yes", "No"])
region = st.selectbox("Region Tempat Tinggal", ["southeast", "southwest", "northwest", "northeast"])


# =========================
# PROSES INPUT MENJADI DATAFRAME SESUAI MODEL
# =========================
def preprocess_input():
    # Raw dictionary dari user
    input_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower()
    }

    df = pd.DataFrame([input_dict])

    # ============================
    # One-Hot Encoding manual
    # ============================
    for col in ohe_columns:
        df[col] = 0

    df[f"smoker_{df['smoker'][0]}"] = 1
    df[f"region_{df['region'][0]}"] = 1

    df.drop(["smoker", "region"], axis=1, inplace=True)

    # ============================
    # Scaling numerik
    # ============================
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # ============================
    # Reorder features agar sesuai model
    # ============================
    df = df.reindex(columns=model_features, fill_value=0)

    return df


# =========================
# PREDIKSI
# =========================
if st.button("Prediksi Biaya Medis"):
    input_df = preprocess_input()

    log_pred = rf_model.predict(input_df)[0]
    final_pred = np.expm1(log_pred)  # kembalikan dari log ke nilai asli

    st.success(f"💰 **Estimasi Biaya Medis: USD {final_pred:,.2f}**")


    st.subheader("📊 Data yang Anda Masukkan")
    st.write(input_df)


st.write("---")
st.caption("Dibuat untuk proyek Machine Learning: Prediksi Biaya Medis Personal")
