import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load model dan data
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("Folds5x2_pp.xlsx")
    return df

model = load_model()
df = load_data()

X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

y_pred_all = model.predict(X)
r2_val = r2_score(y, y_pred_all)
mae_val = mean_absolute_error(y, y_pred_all)
rmse_val = mean_squared_error(y, y_pred_all, squared=False)

# Fungsi rekomendasi
def get_ccpp_recommendation(pe):
    if pe < 430:
        return "⚠️ Daya rendah. Cek sistem pendingin & tekanan udara masuk. Optimasi suhu ambient."
    elif 430 <= pe <= 470:
        return "✅ Daya normal. Sistem berjalan efisien. Lanjutkan monitoring berkala."
    else:
        return "🔥 Daya tinggi. Waspada beban berlebih. Periksa turbin dan pasokan bahan bakar."

# Sidebar navigasi
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih halaman:", 
                        ("Landing Page", "Evaluasi Model", "Transparansi Model", "Prediksi", "Simpan & Unduh", "Info Model"))

if page == "Landing Page":
    st.title("🔌 Prediksi Daya Listrik Pembangkit Listrik Siklus Gabungan (CCPP)")
    st.markdown("""
    Aplikasi ini membantu operator memprediksi output daya CCPP secara akurat berdasarkan kondisi lingkungan seperti suhu, tekanan, kelembapan, dan vakum.
    
    **Pilih halaman di sidebar untuk mulai menggunakan aplikasi.**
    """)
    st.markdown("""
    **Tombol:**
    - [Lihat Evaluasi Model] → pilih halaman 'Evaluasi Model'
    - [Coba Prediksi Langsung] → pilih halaman 'Prediksi'
    """)
    
elif page == "Evaluasi Model":
    st.header("Evaluasi Model")
    st.markdown("Model diuji menggunakan data historis selama 1 tahun.")
    st.markdown(f"- R² Score: **{r2_val:.4f}**")
    st.markdown(f"- MAE (Mean Absolute Error): **{mae_val:.4f}**")
    st.markdown(f"- RMSE (Root Mean Squared Error): **{rmse_val:.4f}**")
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(y, y_pred_all, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Nilai Aktual PE (MW)")
    ax.set_ylabel("Nilai Prediksi PE (MW)")
    ax.set_title("Perbandingan Aktual vs Prediksi")
    st.pyplot(fig)
    
elif page == "Transparansi Model":
    st.header("Transparansi Model")
    st.markdown("Berikut adalah fitur penting yang memengaruhi prediksi model berdasarkan **feature importance**:")
    
    # Feature importance
    importances = model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'fitur': features, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='importance', y='fitur', data=fi_df, ax=ax, palette='viridis')
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    
    st.markdown("""
    Faktor utama yang mempengaruhi keluaran daya adalah suhu ambien (AT) dan vakum cerobong (V), sesuai dengan kondisi fisik turbin.
    """)
    
elif page == "Prediksi":
    st.header("Input Kondisi Lingkungan untuk Prediksi")
    
    at = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)
    
    if st.button("Prediksi Output Daya"):
        X_new = np.array([[at, v, ap, rh]])
        pred_pe = model.predict(X_new)[0]
        
        st.subheader("💡 Hasil Prediksi")
        st.write(f"Prediksi Net Hourly Electrical Energy Output (PE): **{pred_pe:.2f} MW**")
        
        rekomendasi = get_ccpp_recommendation(pred_pe)
        st.info(rekomendasi)
        
        # Cek apakah data ada di dataset
        df_match = df[
            (df['AT'].round(2) == round(at, 2)) &
            (df['V'].round(2) == round(v, 2)) &
            (df['AP'].round(2) == round(ap, 2)) &
            (df['RH'].round(2) == round(rh, 2))
        ]
        
        if not df_match.empty:
            actual_pe = df_match['PE'].values[0]
            error = abs(actual_pe - pred_pe)
            st.success(f"Nilai aktual PE dari dataset: **{actual_pe:.2f} MW**")
            st.info(f"Selisih absolut prediksi vs aktual: **{error:.2f} MW**")
        else:
            st.warning("Data input ini tidak ditemukan dalam dataset asli, nilai aktual tidak tersedia.")
        
        # Visualisasi
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['PE'], bins=50, kde=True, ax=ax, color='skyblue')
        ax.axvline(pred_pe, color='red', linestyle='--', label='Prediksi Anda')
        ax.set_title("Distribusi Output Energi Listrik (PE)")
        ax.set_xlabel("PE (MW)")
        ax.legend()
        st.pyplot(fig)

elif page == "Simpan & Unduh":
    st.header("Simpan dan Unduh Laporan Prediksi")
    st.markdown("Fitur ini akan segera hadir.")
    st.info("Fitur simpan prediksi, unduh laporan CSV, dan kirim ke email operator akan dikembangkan selanjutnya.")

elif page == "Info Model":
    st.header("Informasi Model dan Kontak")
    st.markdown("""
    - Pembuat: **faizah-ra**
    - Data pelatihan: 2020-2023 dari pembangkit X
    - Model: Gradient Boosting Regressor
    - Terakhir diperbarui: Mei 2025
    
    Jika ada pertanyaan, silakan hubungi pembuat aplikasi melalui [GitHub](https://github.com/faizah-ra)
    """)
