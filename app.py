import streamlit as st
import numpy as np
import pickle

# Load model Gradient Boosting
with open('model_gradient_boosting.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi
st.title("Prediksi Daya Listrik CCPP")
st.subheader("Combined Cycle Power Plant Output Prediction")
st.markdown("Masukkan data lingkungan untuk memprediksi output daya (MW)")

# Input fitur dari user
temperature = st.number_input('Suhu Udara (°C)', min_value=0.0, max_value=50.0, value=25.0)
exhaust_vacuum = st.number_input('Vakum (cm Hg)', min_value=20.0, max_value=100.0, value=60.0)
ambient_pressure = st.number_input('Tekanan Udara (mbar)', min_value=900.0, max_value=1100.0, value=1010.0)
relative_humidity = st.number_input('Kelembaban Relatif (%)', min_value=0.0, max_value=100.0, value=60.0)

# Tombol prediksi
if st.button("Prediksi Daya Listrik"):
    # Membentuk input array
    input_data = np.array([[temperature, exhaust_vacuum, ambient_pressure, relative_humidity]])
    
    # Melakukan prediksi
    prediction = model.predict(input_data)[0]
    
    # Menampilkan hasil
    st.success(f"Perkiraan Output Daya: {prediction:.2f} MW")

# Footer
st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh Fay • Model: Gradient Boosting Regressor")
