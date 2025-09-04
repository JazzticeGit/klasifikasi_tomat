import joblib
import pandas as pd
import streamlit as st

model = joblib.load("model_klasifikas_tomat.joblib")
scaler = joblib.load("scaler_klasifikas_tomat.joblib")

st.title("🍅 Klasifikasi Tomat ")
st.markdown("Aplikasi Machine Learning untuk klasifikasi tomat termasuk kategori **Ekspor, Lokal Premium, atau Industri** 🍅")

berat = st.slider(" Berat Tomat", 50, 200, 80)
kekenyalan = st.slider(" Kekenyalan", 2.0, 10.0, 4.2)
kadar_gula = st.slider(" Kadar Gula", 1.0, 10.0, 5.3)
tebal_kulit = st.slider(" Tebal Kulit", 0.1, 1.0, 0.7)

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame(
        [[berat, kekenyalan, kadar_gula, tebal_kulit]],
        columns=["berat", "kekenyalan", "kadar_gula", "tebal_kulit"]
    )
    data_baru_scaled = scaler.transform(data_baru)
    kelas_prediksi = model.predict(data_baru_scaled)[0]
    probabilitas = model.predict_proba(data_baru_scaled)[0]
    presentase = max(probabilitas)

    st.success(f" Model memprediksi: **{kelas_prediksi}** ")
    st.write(" Probabilitas per kelas:")
    for kelas, prob in zip(model.classes_, probabilitas):
        st.write(f"- {kelas}: {prob*100:.2f}%")
    st.info(f" Tingkat keyakinan tertinggi: {presentase*100:.2f}%")
