import streamlit as st
import pandas as pd
import joblib
from preprocessing_func import *
from prediction_funct import *

# --- Set Layout dan Tampilan Header ---
st.set_page_config(layout="wide", page_title="Aplikasi Prediksi Performa Mahasiswa", page_icon="🎓")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/student_logo.jpg", width=130)
with col2:
    st.title('🎓 Aplikasi Prediksi Performa Mahasiswa')
    st.caption("Prototype untuk memprediksi status mahasiswa berdasarkan data akademik dan demografis")

data = pd.DataFrame()

def safe_selectbox(label, encoder, index=1):
    options = encoder.classes_
    safe_index = index if len(options) > index else 0
    return st.selectbox(label=label, options=options, index=safe_index)

# ================================================================================================================
# Bagian 1: Informasi Pribadi dan Akademik
# ================================================================================================================
st.markdown("### 📌 Informasi Pribadi dan Akademik")
col1, col2, col3, col4 = st.columns(4)
with col1:
    status_perkawinan = safe_selectbox('Status Perkawinan', encoder_Marital_status, 1)
    data["Marital_status"] = [status_perkawinan]

    mode_aplikasi = safe_selectbox('Mode Aplikasi', encoder_Application_mode, 1)
    data["Application_mode"] = [mode_aplikasi]

    urutan_aplikasi = st.number_input('Urutan Aplikasi', value=5)
    data["Application_order"] = urutan_aplikasi

    kursus = safe_selectbox('Program Studi', encoder_Course, 5)
    data["Course"] = kursus

with col2:
    periode_kuliah = safe_selectbox('Periode Kuliah', encoder_Daytime_evening_attendance, 1)
    data["Daytime_evening_attendance"] = [periode_kuliah]

    pendidikan_sebelumnya = safe_selectbox('Pendidikan Sebelumnya', encoder_Previous_qualification, 1)
    data["Previous_qualification"] = [pendidikan_sebelumnya]

    nilai_pendidikan_sebelumnya = st.number_input('Nilai Pendidikan Sebelumnya', value=122.0)
    data["Previous_qualification_grade"] = nilai_pendidikan_sebelumnya

    kewarganegaraan = safe_selectbox('Kewarganegaraan', encoder_Nacionality, 1)
    data["Nacionality"] = kewarganegaraan

# ================================================================================================================
# Bagian 2: Informasi Orang Tua
# ================================================================================================================
st.markdown("### 👪 Informasi Orang Tua")
col1, col2, col3, col4 = st.columns(4)
with col1:
    pendidikan_ibu = safe_selectbox('Pendidikan Ibu', encoder_Mothers_qualification, 1)
    data["Mothers_qualification"] = [pendidikan_ibu]

    pendidikan_ayah = safe_selectbox('Pendidikan Ayah', encoder_Fathers_qualification, 1)
    data["Fathers_qualification"] = [pendidikan_ayah]

with col2:
    pekerjaan_ibu = safe_selectbox('Pekerjaan Ibu', encoder_Mothers_occupation, 5)
    data["Mothers_occupation"] = pekerjaan_ibu

    pekerjaan_ayah = safe_selectbox('Pekerjaan Ayah', encoder_Fathers_occupation, 1)
    data["Fathers_occupation"] = [pekerjaan_ayah]

# ================================================================================================================
# Bagian 3: Data Akademik Semester
# ================================================================================================================
st.markdown("### 📚 Data Akademik Semester")
for semester in ["1st", "2nd"]:
    st.markdown(f"#### Semester {semester}")
    col1, col2, col3 = st.columns(3)
    with col1:
        data[f"Curricular_units_{semester}_sem_credited"] = st.number_input(f'SKS Diambil Semester {semester}', value=0)
        data[f"Curricular_units_{semester}_sem_enrolled"] = st.number_input(f'Mata Kuliah Diambil Semester {semester}', value=6)
        data[f"Curricular_units_{semester}_sem_evaluations"] = st.number_input(f'Evaluasi Semester {semester}', value=6)

    with col2:
        data[f"Curricular_units_{semester}_sem_approved"] = st.number_input(f'SKS Lulus Semester {semester}', value=6.0)
        data[f"Curricular_units_{semester}_sem_grade"] = st.number_input(f'IPK Semester {semester}', value=14.0)
        data[f"Curricular_units_{semester}_sem_without_evaluations"] = st.number_input(f'Tanpa Evaluasi Semester {semester}', value=0)

# ================================================================================================================
# Bagian 4: Data Ekonomi Makro
# ================================================================================================================
st.markdown("### 💰 Data Ekonomi Makro")
col1, col2, col3 = st.columns(3)
with col1:
    data["Unemployment_rate"] = st.number_input('Tingkat Pengangguran (%)', value=10.80)
with col2:
    data["Inflation_rate"] = st.number_input('Tingkat Inflasi (%)', value=1.40)
with col3:
    data["GDP"] = st.number_input('PDB (Produk Domestik Bruto)', value=1.74)

# ================================================================================================================
# Tombol Prediksi
# ================================================================================================================
if st.button('Prediksi'):
    st.markdown("### 🔍 Hasil Prediksi")
    new_data = data_preprocessing(data=data)
    new_data = new_data[model.feature_names_in_]

    with st.expander("Lihat Data yang Sudah Diproses"):
        st.dataframe(new_data)

    hasil_prediksi = prediction(new_data)
    if hasil_prediksi == "Enrolled":
        st.success("✅ Mahasiswa Diprediksi MASIH AKTIF")
    else:
        st.error("❌ Mahasiswa Diprediksi TIDAK AKTIF")

