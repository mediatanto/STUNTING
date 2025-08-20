import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("XGBoost_Tanpa_Smote_8020_model.pkl")

st.title("Deteksi Dini Risiko Stunting Pada Anak")

# Input fitur
anak_ke = st.number_input("Anak ke-", min_value=1, step=1)

# Label jenis kelamin
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
jenis_kelamin_label = 1 if jenis_kelamin == "Laki-laki" else 0

# Label IMD
imd = st.selectbox("Inisiasi Menyusu Dini (IMD)", ["Ya", "Tidak"])
imd_label = 1 if imd == "Ya" else 2

# Fitur lainnya
panjang_lahir = st.number_input("Panjang Lahir (cm)", min_value=30.0, max_value=60.0, step=0.1)
berat_lahir = st.number_input("Berat Lahir (kg)", min_value=1.0, max_value=6.0, step=0.1)
lingkar_kepala_lahir = st.number_input("Lingkar Kepala (cm)", min_value=25.0, max_value=49.0, step=0.1)
usia_ibu_hamil = st.number_input("Usia Ibu Saat Hamil", min_value=15, max_value=50, step=1)

# Buat DataFrame dari input user
input_data = pd.DataFrame({
    'anak_ke': [anak_ke],
    'jenis_kelamin': [jenis_kelamin_label],
    'imd': [imd_label],
    'panjang_lahir': [panjang_lahir],
    'berat_lahir': [berat_lahir],
    'lingkar_kepala_lahir': [lingkar_kepala_lahir],
    'usia_ibu_hamil': [usia_ibu_hamil]
})

# Tombol prediksi
if st.button("Prediksi Risiko"):
    prob_berisiko = model.predict_proba(input_data)[0][1]
    prediksi = 1 if prob_berisiko >= 0.4 else 0

    st.subheader("Hasil Prediksi")
    if prediksi == 1:
        st.error("⚠️ Anak Berisiko Stunting")
        st.markdown("---")
        st.markdown("### Catatan")
        st.write(
            "Segera lakukan pemeriksaan kesehatan anak ke fasilitas kesehatan terdekat untuk mendapatkan penanganan dini. "
            "Perhatikan asupan gizi, pola makan, dan kebersihan lingkungan."
        )
    else:
        st.success("✅ Anak Tidak Berisiko Stunting")
        st.markdown("---")
        st.markdown("### Catatan")
        st.write(
            "Tetap pantau pertumbuhan anak secara berkala dan berikan asupan gizi seimbang agar terhindar dari risiko stunting di masa depan."
        )