import streamlit as st
import pandas as pd
import pickle
import numpy as np


st.title('Loan Approvement')
st.write('Masukkan data peminjaman')

@st.cache_resource
def load_model():
    with open('Loan_model_pipeline_2.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)
    return model_pipeline

try:
    model_pipeline = load_model()
    st.success('Model berhasil dimuat!')
except Exception as e:
    st.error(f'Gagal memuat model: {e}')
    st.stop()

# Form input data
st.subheader('Data Peminjam')

# Kolom input untuk fitur numerik
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input('Usia', min_value=18, max_value=100, key="person_age")
    person_income = st.number_input('Pendapatan Tahunan', min_value=0, max_value=1000000, key="person_income")
    person_emp_exp = st.number_input('Pengalaman Kerja (tahun)', min_value=0, max_value=50, key="person_emp_exp")
    loan_amnt = st.number_input('Jumlah Pinjaman', min_value=500, max_value=50000, key="loan_amnt")
    loan_int_rate = st.number_input('Suku Bunga Pinjaman (%)', min_value=1.0, max_value=30.0, key="loan_int_rate", step=0.1)

with col2:
    loan_percent_income = st.number_input('Persentase Pinjaman terhadap Pendapatan', min_value=0.0, max_value=1.0, key="loan_percent_income", step=0.01)
    cb_person_cred_hist_length = st.number_input('Panjang Sejarah Kredit (tahun)', min_value=0, max_value=50, key="cb_person_cred_hist_length")
    credit_score = st.number_input('Skor Kredit', min_value=300, max_value=850, key="credit_score")

# Kolom input untuk fitur kategorikal
st.subheader('Informasi Tambahan')

col3, col4 = st.columns(2)

with col3:
    person_gender = st.selectbox('Jenis Kelamin', ['female', 'male'], key="person_gender")
    person_education = st.selectbox('Pendidikan', ['Master', 'High School', 'Bachelor', 'Associate', 'Doctorate'], key="person_education")
    person_home_ownership = st.selectbox('Status Kepemilikan Rumah', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'], key="person_home_ownership")

with col4:
    loan_intent = st.selectbox('Tujuan Pinjaman', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], key="loan_intent")
    previous_loan_defaults = st.selectbox('Riwayat Gagal Bayar Pinjaman Sebelumnya', ['No', 'Yes'], key="previous_loan_defaults")

# Tombol prediksi
if st.button('Prediksi Status Pinjaman'):
    # Membuat DataFrame dari input
    data = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults
    }
    
    # Mengubah data menjadi DataFrame
    input_df = pd.DataFrame([data])
    
    # Menampilkan data yang dimasukkan
    st.subheader('Data yang Dimasukkan:')
    st.write(input_df)
    
    # Melakukan prediksi
    try:
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)
        
        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi:')
        
        if prediction[0] == 0:
            st.error('Status Pinjaman: **BERISIKO GAGAL BAYAR**')
            st.write(f'Probabilitas Gagal Bayar: {prediction_proba[0][0]:.2%}')
        elif prediction[0] == 1:
            st.success('Status Pinjaman: **KEMUNGKINAN LUNAS**')
            st.write(f'Probabilitas Lunas: {prediction_proba[0][1]:.2%}')
        
    except Exception as e:
        st.error(f'Terjadi kesalahan saat melakukan prediksi: {e}')

st.sidebar.header('Test Cases')

# Fungsi untuk mengatur test case
def set_test_case_1():
    # Data untuk test case 1
    st.session_state.person_age = 24
    st.session_state.person_gender = 'female'
    st.session_state.person_education = 'Associate'
    st.session_state.person_income = 100684
    st.session_state.person_emp_exp = 3
    st.session_state.person_home_ownership = 'RENT'
    st.session_state.loan_amnt = 35000
    st.session_state.loan_intent = 'PERSONAL'
    st.session_state.loan_int_rate = 8.9
    st.session_state.loan_percent_income = 0.35
    st.session_state.cb_person_cred_hist_length = 2
    st.session_state.credit_score = 544
    st.session_state.previous_loan_defaults = 'No'
    st.session_state.test_case_clicked = True

def set_test_case_2():
    # Data untuk test case 2
    st.session_state.person_age = 30.0
    st.session_state.person_gender = 'male'
    st.session_state.person_education = 'Bachelor'
    st.session_state.person_income = 90000.0
    st.session_state.person_emp_exp = 5
    st.session_state.person_home_ownership = 'MORTGAGE'
    st.session_state.loan_amnt = 15000.0
    st.session_state.loan_intent = 'DEBTCONSOLIDATION'
    st.session_state.loan_int_rate = 11.5
    st.session_state.loan_percent_income = 0.17
    st.session_state.cb_person_cred_hist_length = 6.0
    st.session_state.credit_score = 710
    st.session_state.previous_loan_defaults = 'No'
    st.session_state.test_case_clicked = True

# Tombol test case dengan callback functions
st.sidebar.button('Test Case 1 - Kemungkinan Lunas', on_click=set_test_case_1)
st.sidebar.button('Test Case 2 - Berisiko Gagal Bayar', on_click=set_test_case_2)