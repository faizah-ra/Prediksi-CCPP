# Streamlit App: Trust-Oriented User Flow untuk Prediksi Daya CCPP

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Split fitur dan target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']
y_pred_all = model.predict(X)

# Hitung metrik evaluasi
r2_val = r2_score(y, y_pred_all)
mae_val = mean_absolute_error(y, y_pred_all)
rmse_val = mean_squared_error(y, y_pred_all, squared=False)

# Konfigurasi halaman utama
st.set_page_config(page_title="Prediksi Daya CCPP", layout="centered")

# Sidebar navigasi
st.sidebar.title("🔧 Navigasi Aplikasi")
halaman = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda", "📈 Evaluasi Model", "🧠 Transparansi Model", "🔍 Prediksi Langsung"])

if halaman == "🏠 Beranda":
    st.title("🔌 Prediksi Daya Listrik - CCPP")
    st.markdown("""
    Aplikasi ini membantu operator memprediksi **Net Hourly Electrical Energy Output** dari Combined Cycle Power Plant (CCPP) berdasarkan kondisi lingkungan:

    - Ambient Temperature (AT)
    - Exhaust Vacuum (V)
    - Ambient Pressure (AP)
    - Relative Humidity (RH)

    👉 Gunakan menu di samping untuk mengeksplorasi fitur aplikasi.
    """)

elif halaman == "📈 Evaluasi Model":
    st.title("📊 Evaluasi Model Prediksi")
    st.metric("R² Score", f"{r2_val:.4f}")
    st.metric("MAE", f"{mae_val:.2f} MW")
    st.metric("RMSE", f"{rmse_val:.2f} MW")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y, y_pred_all, alpha=0.5, color='blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Aktual vs Prediksi Output Energi")
    st.pyplot(fig)

    st.info("Model diuji menggunakan data historis dari CCPP dan menunjukkan kinerja yang baik dengan R² tinggi.")

elif halaman == "🧠 Transparansi Model":
    st.title("🔍 Transparansi Model")
    feature_importance = model.feature_importances_
    fitur = ['AT', 'V', 'AP', 'RH']
    importance_df = pd.DataFrame({'Fitur': fitur, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Fitur', palette='viridis', ax=ax)
    ax.set_title("Feature Importance Model Gradient Boosting")
    st.pyplot(fig)

    st.write("""
    Faktor paling berpengaruh dalam model ini adalah **Ambient Temperature** dan **Exhaust Vacuum**, 
    yang secara fisik memengaruhi performa turbin gas.
    """)

elif halaman == "🔍 Prediksi Langsung":
    st.title("🔮 Prediksi Output Daya CCPP")

    st.subheader("Masukkan Kondisi Lingkungan")
    at = st.number_input("Ambient Temperature (AT) °C", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    v = st.number_input("Exhaust Vacuum (V) cm Hg", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
    ap = st.number_input("Ambient Pressure (AP) mbar", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    rh = st.number_input("Relative Humidity (RH) %", min_value=10.0, max_value=100.0, value=60.0, step=0.1)

    if st.button("Prediksi Daya"):
        X_new = np.array([[at, v, ap, rh]])
        pred_pe = model.predict(X_new)[0]

        st.subheader("💡 Hasil Prediksi")
        st.write(f"**Prediksi PE:** `{pred_pe:.2f} MW`")
        st.write(f"**R² Score Model:** `{r2_val:.4f}`")

        def get_ccpp_recommendation(pe):
            if pe < 430:
                return "⚠️ Daya rendah. Cek sistem pendingin & tekanan udara masuk."
            elif 430 <= pe <= 470:
                return "✅ Daya normal. Sistem efisien."
            else:
                return "🔥 Daya tinggi. Waspadai beban berlebih."

        st.subheader("📌 Rekomendasi Operasional")
        st.info(get_ccpp_recommendation(pred_pe))

        df_match = df[
            (df['AT'].round(2) == round(at, 2)) &
            (df['V'].round(2) == round(v, 2)) &
            (df['AP'].round(2) == round(ap, 2)) &
            (df['RH'].round(2) == round(rh, 2))
        ]
        if not df_match.empty:
            actual_pe = df_match['PE'].values[0]
            error = abs(actual_pe - pred_pe)
            st.success(f"🎯 Nilai aktual: **{actual_pe:.2f} MW**")
            st.info(f"Selisih prediksi: **{error:.2f} MW**")

        st.subheader("Distribusi Data PE")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['PE'], bins=50, kde=True, ax=ax, color='skyblue')
        ax.axvline(pred_pe, color='red', linestyle='--', label='Prediksi Anda')
        ax.set_title("Distribusi Output Energi Listrik (PE)")
        ax.set_xlabel("PE (MW)")
        ax.legend()
        st.pyplot(fig)

    st.download_button(
        label="⬇️ Unduh Hasil Prediksi",
        data=pd.DataFrame({"AT": [at], "V": [v], "AP": [ap], "RH": [rh], "Prediksi_PE": [pred_pe]}).to_csv(index=False),
        file_name="prediksi_ccpp.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Model Gradient Boosting dilatih menggunakan data UCI CCPP (2011–2014). Aplikasi oleh [faizah-ra](https://github.com/faizah-ra)")
