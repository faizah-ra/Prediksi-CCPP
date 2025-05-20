import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Fungsi Load Model dan Data dengan Cache untuk efisiensi ---
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("Folds5x2_pp.xlsx")

# --- Fungsi rekomendasi operasional berdasarkan nilai prediksi PE ---
def get_ccpp_recommendation(pred_pe):
    if pred_pe >= 460:
        return "Output daya sangat optimal. Pertahankan kondisi operasi saat ini."
    elif 440 <= pred_pe < 460:
        return "Output daya baik, namun lakukan pemantauan lebih intensif."
    elif 420 <= pred_pe < 440:
        return "Output daya menurun, periksa kondisi lingkungan dan peralatan."
    else:
        return "Output daya rendah. Segera lakukan inspeksi dan perawatan mesin."

# --- Load model dan data ---
model = load_model()
df = load_data()

# --- Split fitur dan target ---
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# --- Prediksi seluruh data untuk evaluasi ---
y_pred_all = model.predict(X)

# --- Hitung metrik evaluasi ---
mae = mean_absolute_error(y, y_pred_all)
mse = mean_squared_error(y, y_pred_all)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_all)

# --- Feature importance (hardcoded) ---
feature_importance = {
    "AT": 0.9105252325803225,
    "V": 0.058625172782309824,
    "AP": 0.017573154012486866,
    "RH": 0.013276440624880719
}

# --- URL gambar SHAP ---
SHAP_IMAGE_URL = "https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/655e3c655cff9f581ba13e5fdaf27aff11b3b8e7/shap_beeswarm.png"

# --- Inisialisasi session_state untuk page jika belum ada ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Landing Page"

# --- Sidebar navigasi ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih halaman:",
    ["🏠 Landing Page", "📊 Evaluasi Model", "🔍 Transparansi Model", "⚡ Prediksi Daya", "💾 Simpan & Unduh", "ℹ️ Info Model"],
    index=["🏠 Landing Page", "📊 Evaluasi Model", "🔍 Transparansi Model", "⚡ Prediksi Daya", "💾 Simpan & Unduh", "ℹ️ Info Model"].index(st.session_state.page),
    key="page"
)
st.session_state.page = page

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

# --- Halaman 2: Evaluasi Model ---
elif page == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model Prediksi Daya CCPP")
    st.markdown("Berikut ini adalah hasil evaluasi model Gradient Boosting pada dataset keseluruhan:")
    st.write(f"- Mean Absolute Error (MAE): **{mae:.3f}**")
    st.write(f"- Mean Squared Error (MSE): **{mse:.3f}**")
    st.write(f"- Root Mean Squared Error (RMSE): **{rmse:.3f}**")
    st.write(f"- Koefisien Determinasi (R² Score): **{r2:.3f}**")

    st.subheader("Visualisasi Prediksi vs Aktual")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred_all, alpha=0.5, color='blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Nilai Aktual PE (MW)")
    ax.set_ylabel("Nilai Prediksi PE (MW)")
    ax.set_title("Plot Prediksi vs Nilai Aktual")
    st.pyplot(fig)

# --- Halaman 3: Transparansi Model ---
elif page == "🔍 Transparansi Model":
    st.title("🔍 Transparansi Model - Feature Importance & SHAP")
    st.markdown("Model Gradient Boosting sangat dipengaruhi oleh fitur berikut:")
    importance_df = pd.DataFrame({
        "Fitur": list(feature_importance.keys()),
        "Importance": list(feature_importance.values())
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Fitur"))

    st.markdown("---")
    st.markdown("Visualisasi SHAP (SHapley Additive exPlanations) menunjukkan pengaruh fitur terhadap prediksi:")
    st.image(SHAP_IMAGE_URL, caption="SHAP Beeswarm Plot", use_column_width=True)

# --- Halaman 4: Prediksi Daya ---
elif page == "⚡ Prediksi Daya":
    st.title("⚡ Prediksi Daya Listrik")
    st.header("Masukkan Kondisi Lingkungan untuk Prediksi")

    at = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)

    if st.button("Prediksi"):
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
        at_s = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        v_s = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
        ap_s = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
        rh_s = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)

        submitted = st.form_submit_button("Simpan & Unduh")

        if submitted:
            X_new_s = np.array([[at_s, v_s, ap_s, rh_s]])
            pred_pe_s = model.predict(X_new_s)[0]
            rekom_s = get_ccpp_recommendation(pred_pe_s)

            # Buat DataFrame laporan
            laporan_df = pd.DataFrame({
                "Parameter": ["Ambient Temperature (AT) °C", "Exhaust Vacuum (V) cm Hg", "Ambient Pressure (AP) mbar", "Relative Humidity (RH) %", "Prediksi PE (MW)", "Rekomendasi Operasional"],
                "Nilai": [at_s, v_s, ap_s, rh_s, round(pred_pe_s, 2), rekom_s]
            })

            st.success("Laporan prediksi berhasil dibuat.")
            st.dataframe(laporan_df)

            # Simpan ke Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                laporan_df.to_excel(writer, index=False, sheet_name='Laporan Prediksi')
                writer.save()
                processed_data = output.getvalue()

            st.download_button(
                label="⬇️ Unduh Laporan Excel",
                data=processed_data,
                file_name="Laporan_Prediksi_CCPP.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- Halaman 6: Info Model ---
elif page == "ℹ️ Info Model":
    st.title("ℹ️ Informasi Model Gradient Boosting")
    st.markdown("""
    Model Gradient Boosting yang digunakan memiliki keunggulan:
    - Mampu menangani data numerik dan outlier dengan baik.
    - Memberikan prediksi yang akurat dengan RMSE rendah.
    - Feature importance memberikan insight terhadap variabel paling berpengaruh.
    
    **Referensi:**
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [https://doi.org/10.1145/2939672.2939785]
    - Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.
    """)

