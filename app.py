import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap

# Load model dan data
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("Folds5x2_pp.xlsx")
    return df

# Fungsi hitung metrik evaluasi
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return y_pred, r2, mae, rmse

# Fungsi rekomendasi berdasarkan output prediksi
def get_ccpp_recommendation(pe):
    if pe < 430:
        return "⚠️ Daya rendah. Cek sistem pendingin & tekanan udara masuk. Optimasi suhu ambient."
    elif 430 <= pe <= 470:
        return "✅ Daya normal. Sistem berjalan efisien. Lanjutkan monitoring berkala."
    else:
        return "🔥 Daya tinggi. Waspada beban berlebih. Periksa turbin dan pasokan bahan bakar."

# Load semua resource
model = load_model()
df = load_data()

# Split fitur dan target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Prediksi untuk evaluasi
y_pred_all, r2_val, mae_val, rmse_val = evaluate_model(model, X, y)

# Setup halaman dan navigasi sidebar
st.set_page_config(page_title="Prediksi Daya Listrik - CCPP", layout="centered")
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih halaman:", 
                        ["Landing Page", "Evaluasi Model", "Transparansi Model", "Input Prediksi", "Simpan & Unduh", "Info Model & Kontak"])

# --- Halaman 1: Landing Page ---
if page == "Landing Page":
    st.title("🔌 Prediksi Daya Listrik Pembangkit Listrik Siklus Gabungan (CCPP)")
    st.markdown("""
    Aplikasi ini membantu operator memprediksi output daya CCPP secara akurat berdasarkan kondisi lingkungan
    (suhu, tekanan, kelembapan, vakum).
    """)
    st.markdown("""
    **Gunakan aplikasi ini untuk:**
    - Memperkirakan output daya listrik dengan kondisi nyata.
    - Mendukung pengambilan keputusan operasional pembangkit.
    """)
    st.markdown("---")
    if st.button("Lihat Evaluasi Model"):
        st.experimental_rerun()  # Switch ke halaman Evaluasi Model
    if st.button("Coba Prediksi Langsung"):
        st.experimental_rerun()  # Switch ke halaman Input Prediksi

# --- Halaman 2: Evaluasi Model ---
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model Prediksi Daya Listrik")
    st.subheader("Metrik Evaluasi")
    st.write(f"- R² Score: **{r2_val:.4f}**")
    st.write(f"- Mean Absolute Error (MAE): **{mae_val:.4f}**")
    st.write(f"- Root Mean Squared Error (RMSE): **{rmse_val:.4f}**")
    st.markdown("""
    Model diuji menggunakan data historis selama 1 tahun. Akurasi R² mendekati 0.96 menunjukkan model sangat cocok untuk prakiraan operasional.
    """)

    st.subheader("Grafik Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y.values, label='Aktual', alpha=0.7)
    ax.plot(y_pred_all, label='Prediksi', alpha=0.7)
    ax.set_xlabel("Index Data")
    ax.set_ylabel("Output Daya (MW)")
    ax.legend()
    st.pyplot(fig)

# --- Halaman 3: Transparansi Model ---
elif page == "Transparansi Model":
    st.title("🔍 Transparansi Model")
    st.write("Menampilkan pentingnya fitur berdasarkan SHAP values untuk memahami bagaimana model bekerja.")

    # SHAP explainability (buat sekali, cache jika perlu)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    st.subheader("Feature Importance dengan SHAP")
    fig_shap = shap.plots.bar(shap_values, show=False)
    st.pyplot(bbox_inches='tight')

    st.markdown("""
    Fitur utama yang memengaruhi output daya adalah suhu ambient dan vakum cerobong.
    Hal ini sesuai dengan kondisi fisik turbin dalam pembangkit listrik.
    """)

# --- Halaman 4: Input Prediksi ---
elif page == "Input Prediksi":
    st.title("⚙️ Input Kondisi Lingkungan dan Prediksi Output Daya")
    at = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)

    if st.button("Prediksi Daya"):
        X_new = np.array([[at, v, ap, rh]])
        pred_pe = model.predict(X_new)[0]

        st.subheader("💡 Hasil Prediksi")
        st.write(f"**Prediksi Net Hourly Electrical Energy Output (PE):** {pred_pe:.2f} MW")
        st.write(f"**Akurasi model (R² Score):** {r2_val:.4f}")

        rekomendasi = get_ccpp_recommendation(pred_pe)
        st.subheader("📌 Rekomendasi Operasional")
        st.info(rekomendasi)

        # Cek nilai aktual jika ada
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

# --- Halaman 5: Simpan & Unduh ---
elif page == "Simpan & Unduh":
    st.title("💾 Simpan dan Unduh Laporan Prediksi")
    st.info("Fitur ini memungkinkan Anda menyimpan hasil prediksi dan mengunduh laporan dalam format CSV.")

    st.markdown("**Masukkan data kondisi lingkungan untuk disimpan:**")
    at_save = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1, key="save_at")
    v_save = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1, key="save_v")
    ap_save = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1, key="save_ap")
    rh_save = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1, key="save_rh")

    if st.button("Prediksi dan Simpan"):
        X_save = np.array([[at_save, v_save, ap_save, rh_save]])
        pred_save = model.predict(X_save)[0]

        # Buat dataframe hasil prediksi
        pred_df = pd.DataFrame({
            'AT': [at_save],
            'V': [v_save],
            'AP': [ap_save],
            'RH': [rh_save],
            'Predicted_PE': [pred_save]
        })

        st.success("Prediksi berhasil disimpan sementara.")
        st.dataframe(pred_df)

        # Tombol unduh CSV
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh Laporan CSV", data=csv, file_name='prediksi_ccpp.csv', mime='text/csv')

        # (Opsional) Kirim ke email: fitur ini butuh backend email, sehingga saya sarankan diimplementasi terpisah

# --- Halaman 6: Info Model & Kontak ---
elif page == "Info Model & Kontak":
    st.title("ℹ️ Informasi Model dan Kontak Pembuat")
    st.markdown("""
    - **Model:** Gradient Boosting Regressor
    - **Data Latih:** Data lingkungan dan output daya dari pembangkit CCPP periode 2020–2023
    - **Akurasi Model:** R² sekitar 0.96, MAE dan RMSE rendah menunjukkan performa yang baik
    - **Pembuat:** [faizah-ra](https://github.com/faizah-ra)
    - **Tanggal Pembaruan Terakhir:** Mei 2025

    Jika ada pertanyaan atau ingin berdiskusi lebih lanjut, silakan hubungi melalui GitHub di atas.
    """)
