import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- HERO SECTION ---
st.title("Pengumpulan Data")

# --- LOAD DATA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../assets/cleaned.csv')
data = pd.read_csv(file_path)

# --- MENAMPILKAN TABEL DATA ---
st.write('Berikut adalah dataset komentar negatif')
columns_to_display = ['original_text', 'source', 'pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']
st.dataframe(data[columns_to_display])

# --- VISUALISASI LABEL ---
# Menghitung total dari kolom tertentu
label_sum = data[['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']].sum().sort_values()
label_sum_df = label_sum.reset_index()
label_sum_df.columns = ['Kategori', 'Total Data']

# Membuat plot pie chart dengan Plotly
fig = px.bar(
    label_sum_df,  # Reset index untuk format yang benar
    x='Kategori',  # Kolom yang berisi nama kategori
    y='Total Data',  # Kolom yang berisi nilai jumlah
    title="Persebaran Label",
    labels={'Kategori': 'Label', 'Total Data': 'Total Data'},  # Mengatur label sumbu
    color_discrete_sequence=px.colors.qualitative.Plotly,  # Skema warna
    text='Total Data'  # Menambahkan jumlah data di atas setiap batang
)

fig.update_traces(textposition='outside')

# Menampilkan plot di Streamlit
st.plotly_chart(fig)

# --- VISUALISASI SOURCE ---
# Menghitung jumlah sumber
source_count = data['source'].value_counts().reset_index()  # Reset index untuk format yang benar
source_count.columns = ['source', 'count']  # Mengatur nama kolom

# Membuat plot pie chart untuk sumber dengan Plotly
fig_source = px.pie(source_count,  # Menggunakan DataFrame yang sudah direset
                     values='count',  # Kolom yang berisi nilai
                     names='source',  # Kolom yang berisi nama sumber
                     title="Persebaran Sumber", 
                     color_discrete_sequence=px.colors.qualitative.Plotly)  # Skema warna

# Menampilkan plot sumber di Streamlit
st.plotly_chart(fig_source)