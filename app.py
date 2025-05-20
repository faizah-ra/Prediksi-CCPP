import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model dan data dengan cache untuk efisiensi
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("Folds5x2_pp.xlsx")

model = load_model()
df = load_data()

# Split fitur dan target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Prediksi untuk seluruh data (evaluasi)
y_pred_all = model.predict(X)

# Hitung metrik evaluasi
mae = mean_absolute_error(y, y_pred_all)
mse = mean_squared_error(y, y_pred_all)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_all)

# Feature importance (hardcoded sesuai permintaan)
feature_importance = {
    "AT": 0.9105252325803225,
    "V": 0.058625172782309824,
    "AP": 0.017573154012486866,
    "RH": 0.013276440624880719
}

# URL gambar SHAP (beeswarm plot)
SHAP_IMAGE_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/shap_beeswarm.png"

# Sidebar untuk navigasi halaman
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", 
                        ["🏠 Landing Page", 
                         "📊 Evaluasi Model", 
                         "🔍 Transparansi Model", 
                         "⚡ Prediksi Daya", 
                         "💾 Simpan & Unduh", 
                         "ℹ️ Info Model"])

# --- Fungsi rekomendasi operasional ---
def get_ccpp_recommendation(pe):
    if pe < 430:
        return "⚠️ Daya rendah. Cek sistem pendingin & tekanan udara masuk. Optimasi suhu ambient."
    elif 430 <= pe <= 470:
        return "✅ Daya normal. Sistem berjalan efisien. Lanjutkan monitoring berkala."
    else:
        return "🔥 Daya tinggi. Waspada beban berlebih. Periksa turbin dan pasokan bahan bakar."

# --- Halaman 1: Landing Page ---
if page == "🏠 Landing Page":
    st.title("🔌 Prediksi Daya Listrik - Pembangkit Listrik Siklus Gabungan (CCPP)")
    st.markdown("""
    Aplikasi ini membantu operator memprediksi output daya CCPP secara akurat berdasarkan kondisi lingkungan:
    - Suhu ambient (AT)
    - Vakum cerobong (V)
    - Tekanan ambient (AP)
    - Kelembapan relatif (RH)
    
    Gunakan prediksi ini untuk memantau performa pembangkit dan mengambil tindakan operasional yang tepat.
    """)
    st.markdown("---")
    st.write("Silakan pilih halaman di sidebar untuk melihat evaluasi model, transparansi, dan melakukan prediksi.")
    st.button("🔎 Coba Prediksi Langsung", on_click=lambda: st.experimental_rerun())

# --- Halaman 2: Evaluasi Model ---
elif page == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model Gradient Boosting Regressor")
    st.markdown("Model diuji menggunakan data historis dari pembangkit CCPP. Berikut metrik evaluasinya:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"{rmse:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")
    
    st.markdown("### Grafik Prediksi vs Aktual")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(y, y_pred_all, alpha=0.3, color='blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Nilai Aktual PE (MW)")
    ax.set_ylabel("Prediksi PE (MW)")
    ax.set_title("Prediksi vs Aktual Output Energi Listrik")
    st.pyplot(fig)
    
    st.info("Model dengan R² mendekati 1 menunjukkan kemampuan prediksi yang sangat baik.")

# --- Halaman 3: Transparansi Model ---
elif page == "🔍 Transparansi Model":
    st.title("🔍 Transparansi Model")
    st.markdown("Berikut ini adalah pentingnya fitur (feature importance) yang memengaruhi prediksi daya listrik:")
    fi_df = pd.DataFrame(feature_importance.items(), columns=["Fitur", "Importance"])
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    st.bar_chart(fi_df.set_index("Fitur"))
    
    st.markdown("""
    **Penjelasan:**
    - Suhu ambient (AT) merupakan faktor paling dominan, berkontribusi sekitar 91%.
    - Vakum cerobong (V), tekanan ambient (AP), dan kelembapan relatif (RH) memiliki pengaruh yang lebih kecil.
    - Hal ini sesuai dengan kondisi fisik turbin yang sangat dipengaruhi suhu dan vakum cerobong.
    """)
    
    st.markdown("### Visualisasi SHAP (SHapley Additive exPlanations)")
    st.image(SHAP_IMAGE_URL, caption="SHAP Beeswarm Plot - Pengaruh Fitur pada Prediksi")

# --- Halaman 4: Prediksi Daya ---
elif page == "⚡ Prediksi Daya":
    st.title("⚡ Prediksi Daya Listrik")
    st.sidebar.header("Input Kondisi Lingkungan")
    at = st.sidebar.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.sidebar.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.sidebar.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.sidebar.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)
    
    X_new = np.array([[at, v, ap, rh]])
    pred_pe = model.predict(X_new)[0]
    
    st.subheader("💡 Hasil Prediksi")
    st.write(f"**Prediksi Net Hourly Electrical Energy Output (PE):** `{pred_pe:.2f} MW`")
    
    rekomendasi = get_ccpp_recommendation(pred_pe)
    st.subheader("📌 Rekomendasi Operasional")
    st.info(rekomendasi)
    
    # Cek kecocokan dengan data asli
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
    
    # Visualisasi distribusi PE
    st.subheader("📊 Distribusi Output Energi Listrik (PE)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(df['PE'], bins=50, kde=True, ax=ax2, color='skyblue')
    ax2.axvline(pred_pe, color='red', linestyle='--', label='Prediksi Anda')
    ax2.set_title("Distribusi Output Energi Listrik (PE)")
    ax2.set_xlabel("PE (MW)")
    ax2.legend()
    st.pyplot(fig2)

# --- Halaman 5: Simpan & Unduh ---
elif page == "💾 Simpan & Unduh":
    st.title("💾 Simpan & Unduh Laporan Prediksi")
    st.markdown("Masukkan data prediksi Anda untuk disimpan dan diunduh.")
    
    with st.form("form_simpan"):
        at = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        v = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
        ap = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
        rh = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)
        submitted = st.form_submit_button("Prediksi & Simpan")
        
        if submitted:
            X_new = np.array([[at, v, ap, rh]])
            pred_pe = model.predict(X_new)[0]
            rekomendasi = get_ccpp_recommendation(pred_pe)
            
            # Simpan data ke dataframe sederhana
            data_to_save = pd.DataFrame({
                "AT": [at],
                "V": [v],
                "AP": [ap],
                "RH": [rh],
                "Prediksi_PE": [pred_pe],
                "Rekomendasi": [rekomendasi]
            })
            
            st.success("Prediksi berhasil dilakukan dan data siap diunduh.")
            st.write(data_to_save)
            
            csv = data_to_save.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Unduh CSV", data=csv, file_name="prediksi_ccpp.csv", mime="text/csv")

# --- Halaman 6: Info Model ---
elif page == "ℹ️ Info Model":
    st.title("ℹ️ Informasi Model dan Pembuat")
    st.markdown("""
    - **Pembuat:** faizah-ra ([GitHub](https://github.com/faizah
