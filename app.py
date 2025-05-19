import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Load model dan data
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("Folds5x2_pp.xlsx")
    return df

# Load semua resource
model = load_model()
df = load_data()

# Split fitur dan target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Hitung skor model untuk informasi
y_pred_all = model.predict(X)
r2_val = r2_score(y, y_pred_all)

# UI Streamlit
st.set_page_config(page_title="Prediksi Daya Listrik - CCPP", layout="centered")
st.title("🔌 Prediksi Daya Listrik Pembangkit Listrik Siklus Gabungan (CCPP)")
st.markdown("Menggunakan **Gradient Boosting Regressor** berdasarkan kondisi lingkungan.")

# Sidebar input
st.sidebar.header("Input Kondisi Lingkungan")
at = st.sidebar.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
v = st.sidebar.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
ap = st.sidebar.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
rh = st.sidebar.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)

# Prediksi
X_new = np.array([[at, v, ap, rh]])
pred_pe = model.predict(X_new)[0]

st.subheader("💡 Hasil Prediksi")
st.write(f"**Prediksi Net Hourly Electrical Energy Output (PE):** `{pred_pe:.2f} MW`")
st.write(f"**Akurasi model (R² Score):** `{r2_val:.4f}`")

# 🔧 Fungsi rekomendasi berdasarkan output prediksi
def get_ccpp_recommendation(pe):
    if pe < 430:
        return "⚠️ Daya rendah. Cek sistem pendingin & tekanan udara masuk. Optimasi suhu ambient."
    elif 430 <= pe <= 470:
        return "✅ Daya normal. Sistem berjalan efisien. Lanjutkan monitoring berkala."
    else:
        return "🔥 Daya tinggi. Waspada beban berlebih. Periksa turbin dan pasokan bahan bakar."

# Tampilkan rekomendasi
st.subheader("📌 Rekomendasi Operasional")
rekomendasi = get_ccpp_recommendation(pred_pe)
st.info(rekomendasi)

# Cek apakah input pernah ada di data asli
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

# Visualisasi
st.subheader("📊 Visualisasi Data PE")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df['PE'], bins=50, kde=True, ax=ax, color='skyblue')
ax.axvline(pred_pe, color='red', linestyle='--', label='Prediksi Anda')
ax.set_title("Distribusi Output Energi Listrik (PE)")
ax.set_xlabel("PE (MW)")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.caption("Sumber data: UCI CCPP | Dibuat oleh [faizah-ra](https://github.com/faizah-ra)")
