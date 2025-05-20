import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Raw URLs dari GitHub
DATASET_URL = "https://github.com/faizah-ra/Prediksi-CCPP/raw/main/Folds5x2_pp.xlsx"
MODEL_URL = "https://github.com/faizah-ra/Prediksi-CCPP/raw/main/model_gradient_boosting.pkl"

@st.cache_resource
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    model_bytes = BytesIO(response.content)
    model = joblib.load(model_bytes)
    return model

@st.cache_data
def load_dataset_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    data_bytes = BytesIO(response.content)
    df = pd.read_excel(data_bytes)
    return df

def main():
    st.title("Prediksi Energi dengan Gradient Boosting")

    # Load model dan dataset
    model = load_model_from_url(MODEL_URL)
    df = load_dataset_from_url(DATASET_URL)

    # Tampilkan dataset (opsional)
    if st.checkbox("Tampilkan dataset"):
        st.dataframe(df)

    # Split fitur dan target
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']

    # Prediksi semua data untuk evaluasi model
    y_pred_all = model.predict(X)

    # Hitung metrik evaluasi
    r2_val = r2_score(y, y_pred_all)
    mae_val = mean_absolute_error(y, y_pred_all)
    rmse_val = mean_squared_error(y, y_pred_all, squared=False)

    st.subheader("Evaluasi Model pada Dataset Lengkap")
    st.write(f"R2 Score: {r2_val:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae_val:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")

    # Input fitur untuk prediksi manual dari user
    st.subheader("Prediksi Energi Berdasarkan Input Manual")
    at = st.number_input("Masukkan AT (Temperature)", value=20.0)
    v = st.number_input("Masukkan V (Exhaust Vacuum)", value=40.0)
    ap = st.number_input("Masukkan AP (Ambient Pressure)", value=1010.0)
    rh = st.number_input("Masukkan RH (Relative Humidity)", value=50.0)

    if st.button("Prediksi"):
        input_data = [[at, v, ap, rh]]
        pred = model.predict(input_data)
        st.success(f"Prediksi Energi (PE): {pred[0]:.4f}")

if __name__ == "__main__":
    main()
