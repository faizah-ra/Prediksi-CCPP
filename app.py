import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Output Daya pada Pembangkit Listrik Siklus Gabungan (CCPP) Berdasarkan Faktor Lingkungan Menggunakan Algoritma Gradient Boosting Regression Berbasis Machine Learning
",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
@st.cache_resource
def load_model():
    return joblib.load("model_gradient_boosting.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("Folds5x2_pp.xlsx")

model = load_model()
df = load_data()
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']
y_pred_all = model.predict(X)

# Navigation
page = st.sidebar.radio("Navigasi", ["🏠 Beranda", "🔮 Prediksi", "📈 Evaluasi Model", "📚 Dokumentasi Teknis"])

# Halaman Beranda
if page == "🏠 Beranda":
    st.title("🔌 Prediksi Output Daya CCPP Berbasis Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Sistem Prediksi Daya Listrik Siklus Gabungan
        **Memanfaatkan algoritma Gradient Boosting Regression** untuk memprediksi output daya pembangkit 
        berdasarkan parameter lingkungan:
        
        - 🌡️ Suhu Ambien (AT)
        - 🌀 Vakum Gas Buang (V)
        - 🎚️ Tekanan Ambien (AP)
        - 💧 Kelembaban Relatif (RH)
        """)
        
    with col2:
        st.image("https://raw.githubusercontent.com/faizah-ra/Prediksi-CCPP/main/ccpp-schema.png", 
                caption="Diagram Sistem CCPP")
    
    st.markdown("---")
    
    st.subheader("📊 Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)

# Halaman Prediksi
elif page == "🔮 Prediksi":
    st.title("🔮 Prediksi Real-Time")
    
    with st.expander("⚙️ Parameter Input", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        at = col1.number_input("🌡️ Suhu Ambien (°C)", 0.0, 50.0, 25.0, 0.1)
        v = col2.number_input("🌀 Vakum Gas Buang (cmHg)", 20.0, 100.0, 40.0, 0.1)
        ap = col3.number_input("🎚️ Tekanan Ambien (mbar)", 900.0, 1100.0, 1013.0, 0.1)
        rh = col4.number_input("💧 Kelembaban Relatif (%)", 10.0, 100.0, 60.0, 0.1)

    # Prediksi
    X_new = np.array([[at, v, ap, rh]])
    pred_pe = model.predict(X_new)[0]
    
    # Tampilan hasil prediksi
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Hasil Prediksi")
        st.metric("Net Hourly Electrical Energy Output", f"{pred_pe:.2f} MW", delta_color="off")
        
        # Card informasi tambahan
        st.markdown("""
        <div style="padding:20px;background:#f0f2f6;border-radius:10px">
            <h4>📊 Performa Model</h4>
            <p>R² Score: 0.9861<br>
            MAE: 1.73 MW<br>
            RMSE: 2.45 MW</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📌 Rekomendasi Operasional")
        recommendation = get_detailed_recommendation(pred_pe)
        st.markdown(f"""
        <div style="padding:20px;background:#e6f4ff;border-radius:10px">
            {recommendation}
        </div>
        """, unsafe_allow_html=True)
    
    # Visualisasi interaktif
    st.markdown("---")
    st.subheader("📊 Distribusi Output Daya")
    plot_pe_distribution(pred_pe)

# Halaman Evaluasi Model
elif page == "📈 Evaluasi Model":
    st.title("📊 Evaluasi Performa Model")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Evaluasi Model", "Feature Importance", "Actual vs Predicted", "Validasi Silang"])
    
    with tab1:
        st.image("https://github.com/faizah-ra/Prediksi-CCPP/raw/cd0ba84ef172d12a1e6c49d40a679846794653d9/download%20(1).png",
                caption="Evaluasi Metrik Model")
    
    with tab2:
        st.image("https://github.com/faizah-ra/Prediksi-CCPP/raw/cd0ba84ef172d12a1e6c49d40a679846794653d9/download%20(2).png",
                caption="Feature Importance")
    
    with tab3:
        st.image("https://github.com/faizah-ra/Prediksi-CCPP/raw/cd0ba84ef172d12a1e6c49d40a679846794653d9/download%20(3).png",
                caption="Actual vs Predicted Values")
    
    with tab4:
        st.image("https://github.com/faizah-ra/Prediksi-CCPP/raw/cd0ba84ef172d12a1e6c49d40a679846794653d9/download.png",
                caption="Cross-Validation MAE")

# Halaman Dokumentasi Teknis
elif page == "📚 Dokumentasi Teknis":
    st.title("📚 Dokumentasi Teknis")
    
    st.markdown("""
    ## Arsitektur Sistem Prediksi
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://github.com/faizah-ra/Prediksi-CCPP/raw/main/workflow.png",
                caption="Diagram Alur Sistem Prediksi")
    
    with col2:
        st.markdown("""
        ### Komponen Utama Sistem:
        1. **Data Collection**: Sistem SCADA pembangkit
        2. **Preprocessing**: Normalisasi dan cleaning data
        3. **Model Training**: Gradient Boosting dengan 1000 trees
        4. **Prediction Engine**: API prediksi real-time
        5. **Visualization**: Dashboard monitoring
        """)
    
    st.markdown("---")
    
    st.subheader("Interpretasi Model (SHAP Values)")
    st.image("https://github.com/faizah-ra/Prediksi-CCPP/raw/cd0ba84ef172d12a1e6c49d40a679846794653d9/shap_beeswarm.png",
            caption="SHAP Value Analysis")

# Fungsi tambahan
def get_detailed_recommendation(pe):
    if pe < 420:
        return """
        🔴 **Kondsi Kritis**  
        - Segera lakukan pengecekan sistem pendingin  
        - Verifikasi tekanan udara masuk turbin  
        - Pertimbangkan reduksi beban generator  
        - Monitor suhu bearing turbin  
        """
    elif 420 <= pe < 440:
        return """
        🟡 **Perhatian Khusus**  
        - Optimasi suhu ambient intake  
        - Cek efisiensi heat exchanger  
        - Verifikasi kalibrasi sensor vakum  
        - Lakukan analisis gas buang  
        """
    elif 440 <= pe <= 460:
        return """
        🟢 **Operasi Optimal**  
        - Pertahankan parameter saat ini  
        - Lanjutkan monitoring rutin  
        - Catat fluktuasi parameter setiap jam  
        - Verifikasi konsumsi bahan bakar  
        """
    else:
        return """
        🔥 **Beban Tinggi**  
        - Waspadai overload generator  
        - Monitor suhu exhaust turbin  
        - Verifikasi sistem pendingin darurat  
        - Siapkan prosedur shutdown darurat  
        """

def plot_pe_distribution(pred_value):
    fig = px.histogram(df, x='PE', nbins=50, 
                      title='Distribusi Output Daya Historis',
                      labels={'PE': 'Output Daya (MW)'})
    
    fig.add_vline(x=pred_value, line_dash="dash", line_color="red",
                 annotation_text=f"Prediksi Anda: {pred_value:.2f} MW")
    
    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
        xaxis_title="Output Daya (MW)",
        yaxis_title="Frekuensi"
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px">
    <p>Dikembangkan oleh Tim Operasi CCPP • 
    <a href="https://github.com/faizah-ra/Prediksi-CCPP">Lihat Kode Sumber</a> • 
    Versi 2.1.0</p>
</div>
""", unsafe_allow_html=True)
