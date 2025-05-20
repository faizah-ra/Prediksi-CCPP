import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# URLs data eksternal
METRICS_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/metrics.json"
FEATURE_IMPORTANCE_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/feature_importance.json"
SHAP_IMAGE_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/shap_beeswarm.png"

# Load model & data dengan caching
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("Folds5x2_pp.xlsx")
    return df

@st.cache_data
def load_metrics():
    response = requests.get(METRICS_URL)
    return response.json()

@st.cache_data
def load_feature_importance():
    response = requests.get(FEATURE_IMPORTANCE_URL)
    return response.json()

model = load_model()
df = load_data()
metrics = load_metrics()
feature_importance = load_feature_importance()

# Split fitur dan target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Sidebar navigation
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["🏠 Beranda", 
                         "📊 Evaluasi Model", 
                         "🔍 Transparansi Model", 
                         "⚡ Prediksi Output", 
                         "💾 Simpan & Unduh", 
                         "ℹ️ Tentang Model"])

# Fungsi rekomendasi operasional
def get_ccpp_recommendation(pe):
    if pe < 430:
        return "⚠️ Daya rendah. Cek sistem pendingin & tekanan udara masuk. Optimasi suhu ambient."
    elif 430 <= pe <= 470:
        return "✅ Daya normal. Sistem berjalan efisien. Lanjutkan monitoring berkala."
    else:
        return "🔥 Daya tinggi. Waspada beban berlebih. Periksa turbin dan pasokan bahan bakar."

# Halaman 1: Beranda
if page == "🏠 Beranda":
    st.title("🔌 Prediksi Daya Listrik CCPP")
    st.markdown(
        """
        Aplikasi ini membantu operator memprediksi output daya pembangkit listrik siklus gabungan (CCPP) secara akurat berdasarkan kondisi lingkungan seperti suhu ambient, tekanan, kelembapan, dan vakum.
        
        **Pilih halaman di sidebar untuk:**  
        - Lihat evaluasi model  
        - Pelajari cara kerja model  
        - Lakukan prediksi dengan data terbaru  
        - Simpan dan unduh laporan hasil prediksi  
        - Info pembuat model dan kontak  
        """
    )
    st.markdown("### Mulai")
    col1, col2 = st.columns(2)
    if col1.button("Lihat Evaluasi Model"):
        st.experimental_set_query_params(page="📊 Evaluasi Model")
    if col2.button("Coba Prediksi Langsung"):
        st.experimental_set_query_params(page="⚡ Prediksi Output")

# Halaman 2: Evaluasi Model
elif page == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model Gradient Boosting Regressor")
    st.markdown("Model diuji menggunakan data historis selama 1 tahun.")
    
    # Tampilkan metrik utama
    st.subheader("Metrik Evaluasi")
    st.write(f"- Mean Absolute Error (MAE): **{metrics['MAE']:.4f}**")
    st.write(f"- Root Mean Squared Error (RMSE): **{metrics['RMSE']:.4f}**")
    st.write(f"- Coefficient of Determination (R² Score): **{metrics['R2']:.4f}**")
    
    # Visualisasi Prediksi vs Aktual
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y, y_pred, alpha=0.4, color='royalblue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Output Aktual (PE) MW")
    ax.set_ylabel("Output Prediksi (PE) MW")
    ax.set_title("Perbandingan Output Aktual vs Prediksi")
    st.pyplot(fig)
    
    st.markdown(
        """
        **Penjelasan:**  
        Akurasi R² sebesar 0.96 menunjukkan model ini sangat cocok untuk prakiraan operasional.  
        Scatter plot menunjukkan prediksi mendekati nilai aktual, membuktikan keandalan model.
        """
    )

# Halaman 3: Transparansi Model
elif page == "🔍 Transparansi Model":
    st.title("🔍 Transparansi Model dan Feature Importance")
    
    # Tampilkan feature importance
    st.subheader("Feature Importance")
    fi_df = pd.DataFrame(feature_importance.items(), columns=["Fitur", "Pentingnya"])
    fi_df = fi_df.sort_values(by="Pentingnya", ascending=False)
    st.bar_chart(fi_df.set_index("Fitur"))
    
    # Tampilkan SHAP image
    st.subheader("Visualisasi SHAP (SHapley Additive exPlanations)")
    st.image(SHAP_IMAGE_URL, caption="SHAP Beeswarm Plot - Pengaruh Fitur terhadap Prediksi", use_column_width=True)
    
    st.markdown(
        """
        **Penjelasan:**  
        Suhu ambient (AT) dan vakum cerobong (V) adalah faktor utama yang memengaruhi keluaran daya listrik (PE).  
        Hal ini sesuai dengan kondisi fisik pembangkit dan turbin yang dipantau secara real-time.
        """
    )

# Halaman 4: Prediksi Output
elif page == "⚡ Prediksi Output":
    st.title("⚡ Prediksi Output Daya Listrik")
    st.sidebar.header("Input Kondisi Lingkungan")
    at = st.sidebar.number_input("Ambient Temperature (AT) °C", 0.0, 50.0, 25.0, 0.1)
    v = st.sidebar.number_input("Exhaust Vacuum (V) cm Hg", 20.0, 100.0, 40.0, 0.1)
    ap = st.sidebar.number_input("Ambient Pressure (AP) mbar", 900.0, 1100.0, 1013.0, 0.1)
    rh = st.sidebar.number_input("Relative Humidity (RH) %", 10.0, 100.0, 60.0, 0.1)
    
    # Prediksi
    X_new = np.array([[at, v, ap, rh]])
    pred_pe = model.predict(X_new)[0]
    
    st.subheader("💡 Hasil Prediksi")
    st.write(f"**Prediksi Net Hourly Electrical Energy Output (PE):** `{pred_pe:.2f} MW`")
    
    rekomendasi = get_ccpp_recommendation(pred_pe)
    st.subheader("📌 Rekomendasi Operasional")
    st.info(rekomendasi)
    
    # Cek data aktual dari dataset asli (jika ada)
    df_match = df[
        (df['AT'].round(2) == round(at, 2)) &
        (df['V'].round(2) == round(v, 2)) &
        (df['AP'].round(2) == round(ap, 2)) &
        (df['RH'].round(2) == round(rh, 2))
    ]
    if not df_match.empty:
        actual_pe = df_match['PE'].values[0]
        error = abs(actual_pe - pred_pe)
        st.success(f"🎯 Nilai aktual PE dari dataset: **{actual_pe:.2f} MW**")
        st.info(f"Selisih absolut prediksi vs aktual: **{error:.2f} MW**")
    else:
        st.warning("⚠️ Data input ini tidak ditemukan dalam dataset asli, nilai aktual tidak tersedia.")
    
    # Visualisasi distribusi PE dan prediksi
    st.subheader("📊 Visualisasi Data PE")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['PE'], bins=50, kde=True, ax=ax, color='skyblue')
    ax.axvline(pred_pe, color='red', linestyle='--', label='Prediksi Anda')
    ax.set_title("Distribusi Output Energi Listrik (PE)")
    ax.set_xlabel("PE (MW)")
    ax.legend()
    st.pyplot(fig)

# Halaman 5: Simpan & Unduh
elif page == "💾 Simpan & Unduh":
    st.title("💾 Simpan dan Unduh Laporan Prediksi")
    st.markdown("Fitur ini memungkinkan Anda menyimpan dan mengunduh hasil prediksi untuk pelaporan dan tracking.")
    
    # Form input ulang untuk prediksi agar bisa simpan
    at = st.number_input("Ambient Temperature (AT) °C", 0.0, 50.0, 25.0, 0.1)
    v = st.number_input("Exhaust Vacuum (V) cm Hg", 20.0, 100.0, 40.0, 0.1)
    ap = st.number_input("Ambient Pressure (AP) mbar", 900.0, 1100.0, 1013.0, 0.1)
    rh = st.number_input("Relative Humidity (RH) %", 10.0, 100.0, 60.0, 0.1)
    
    X_new = np.array([[at, v, ap, rh]])
    pred_pe = model.predict(X_new)[0]
    
    # Buat dataframe hasil prediksi
    pred_df = pd.DataFrame({
        "AT": [at],
        "V": [v],
        "AP": [ap],
        "RH": [rh],
        "Prediksi_PE_MW": [pred_pe]
    })
    
    st.write(pred_df)
    
    # Simpan ke csv dan unduh
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Unduh Laporan CSV", csv, "laporan_prediksi.csv", "text/csv")
    
    # Simulasi kirim email (placeholder)
    if st.button("📧 Kirim ke Email Operator"):
        st.success("Email laporan berhasil dikirim ke operator (simulasi).")

# Halaman 6: Tentang Model
elif page == "ℹ️ Tentang Model":
    st.title("ℹ️ Informasi Model dan Kontak")
    st.markdown(
        """
        **Pembuat Model:** Faizah Ra  
        **Tanggal Update Terakhir:** Mei 2025
