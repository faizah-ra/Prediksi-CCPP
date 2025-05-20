import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

# --- Load Data & Assets dari GitHub (bisa diganti dengan local file jika perlu) ---
METRICS_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/metrics.json"
FEATURE_IMPORTANCE_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/feature_importance.json"
SHAP_IMAGE_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/shap_beeswarm.png"

@st.cache_data
def load_json(url):
    return pd.read_json(url)

@st.cache_data
def load_feature_importance(url):
    # File berformat JSON dengan struktur key:value feature:importance
    fi = pd.read_json(url, typ='series')
    return fi.sort_values(ascending=True)

@st.cache_data
def load_image(url):
    return Image.open(url)

# --- Data Loading ---
metrics = load_json(METRICS_URL)
feature_importance = load_feature_importance(FEATURE_IMPORTANCE_URL)
shap_image = load_image(SHAP_IMAGE_URL)

# --- Header ---
st.title("Prediksi Output Daya CCPP dengan Gradient Boosting")
st.markdown("""
Aplikasi ini membantu operator memprediksi output daya Combined Cycle Power Plant (CCPP)
secara akurat berdasarkan kondisi lingkungan seperti suhu, tekanan, kelembapan, dan vakum.
""")

# --- Sidebar Navigation ---
page = st.sidebar.selectbox("Menu", [
    "Landing Page",
    "Evaluasi Model",
    "Transparansi Model",
    "Prediksi Langsung",
    "Simpan & Unduh Laporan",
    "Informasi Model"
])

if page == "Landing Page":
    st.header("Apa Gunanya Alat Ini?")
    st.write("""
    - Membantu operator dan manajer pembangkit memprediksi output daya secara real-time.
    - Berdasarkan data sensor lingkungan yang akurat.
    - Meningkatkan efisiensi dan perencanaan operasional pembangkit.
    """)
    if st.button("Lihat Evaluasi Model"):
        st.experimental_set_query_params(page="Evaluasi Model")
    if st.button("Coba Prediksi Langsung"):
        st.experimental_set_query_params(page="Prediksi Langsung")

elif page == "Evaluasi Model":
    st.header("Evaluasi Akurasi Model")
    st.write("Model diuji menggunakan data historis selama 1 tahun.")
    
    # Metrik utama
    st.subheader("Metrik Evaluasi:")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:.3f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    col3.metric("R²", f"{metrics['R2']:.3f}")

    st.markdown("""
    > Akurasi R² = 0.96 menunjukkan model sangat cocok untuk prakiraan operasional.
    """)

    # Plot Aktual vs Prediksi
    st.subheader("Grafik Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(metrics['actual'], metrics['predicted'], alpha=0.7, color='darkorange', label='Prediksi')
    ax.plot([min(metrics['actual']), max(metrics['actual'])], [min(metrics['actual']), max(metrics['actual'])], '--r', label='Garis Ideal')
    coef = np.polyfit(metrics['actual'], metrics['predicted'], 1)
    reg_line = np.poly1d(coef)
    ax.plot(metrics['actual'], reg_line(metrics['actual']), color='blue', linestyle='-', label='Garis Regresi')
    ax.set_xlabel("Actual PE")
    ax.set_ylabel("Predicted PE")
    ax.set_title("Actual vs Predicted PE (Gradient Boosting)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

elif page == "Transparansi Model":
    st.header("Bagaimana Model Ini Bekerja?")
    st.write("""
    Faktor utama yang mempengaruhi keluaran daya adalah:
    - Suhu ambien
    - Vakum cerobong
    
    Hal ini sesuai dengan kondisi fisik turbin dan lingkungan pembangkit.
    """)

    # Feature Importance Plot
    st.subheader("Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    feature_importance.plot(kind='barh', color='skyblue', ax=ax2)
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature Importance - Gradient Boosting')
    st.pyplot(fig2)

    # SHAP Plot Image
    st.subheader("SHAP Summary Plot")
    st.image(shap_image, caption="SHAP Beeswarm Plot")

elif page == "Prediksi Langsung":
    st.header("Input Data Lingkungan untuk Prediksi Output Daya")
    with st.form("input_form"):
        suhu = st.number_input("Suhu (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.1)
        tekanan = st.number_input("Tekanan (kPa)", min_value=80.0, max_value=120.0, value=101.3, step=0.1)
        kelembapan = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        vakum = st.number_input("Vakum (mbar)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Dummy prediksi (ganti dengan model nyata yang sudah diload)
        # Contoh prediksi sederhana sebagai ilustrasi
        prediksi_output = 370 + 2*(suhu - 25) - 1.5*(tekanan - 101.3) + 0.8*(kelembapan - 50) - 1.2*(vakum - 50)
        st.success(f"Prediksi output daya hari ini: {prediksi_output:.2f} MW")

elif page == "Simpan & Unduh Laporan":
    st.header("Simpan dan Unduh Laporan Prediksi")
    st.write("Fitur ini sedang dalam pengembangan.")
    # Di sini bisa ditambahkan fitur ekspor CSV, simpan JSON, dll.

elif page == "Informasi Model":
    st.header("Informasi Model & Kontak Pembuat")
    st.write("""
    - Model dibuat oleh Tim AI Pembangkit Listrik.
    - Terakhir diperbarui: Mei 2025
    - Data pelatihan: Data pembangkit CCPP tahun 2020–2023.
    - Kontak: ai-team@example.com
    """)

    st.markdown("---")
    st.write("Terima kasih telah menggunakan aplikasi prediksi ini!")
