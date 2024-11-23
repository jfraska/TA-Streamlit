import streamlit as st
import pandas as pd
import os

# --- HERO SECTION ---
st.title("Text Preprocessing")

# --- LOAD DATA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../assets/cleaned.csv')
data = pd.read_csv(file_path)

st.subheader('**Casefolding dan Cleaning**')
st.write('Mengubah seluruh huruf menjadi kecil (lowercase) yang ada pada dokumen dan membersihkan data dari angka, tanda baca, dll.')
clean_display = data[['original_text', 'text_clean']].copy()
clean_display.rename(columns={'original_text': 'Sebelum', 'text_clean': 'Sesudah'}, inplace=True)
st.dataframe(clean_display.head(10))

# st.subheader('**Tokenize**')
# st.write('Memecah teks menjadi kata-kata atau token.')
# st.dataframe(data[['text_clean']].head(10))

st.subheader('**Translate slang words**')
st.write('Mengubah atau menerjemahkan istilah atau frasa yang dianggap tidak formal ke dalam bahasa yang lebih standar atau formal.')
transform_display = data[['text_clean', 'text_transform']].copy()
transform_display.rename(columns={'text_clean': 'Sebelum', 'text_transform': 'Sesudah'}, inplace=True)
st.dataframe(transform_display.head(10))

st.subheader('**Menghilangkan Stopword**')
st.write('Menghilangkan kata-kata umum yang tidak memiliki makna penting.')
stopwords_display = data[['text_transform', 'text_stopword']].copy()
stopwords_display.rename(columns={'text_transform': 'Sebelum', 'text_stopword': 'Sesudah'}, inplace=True)
st.dataframe(stopwords_display.head(10))

st.subheader('**Stemming**')
st.write('Mengubah kata-kata ke bentuk dasarnya.')
stemming_display = data[['text_stopword', 'text_stemming']].copy()
stemming_display.rename(columns={'text_stopword': 'Sebelum', 'text_stemming': 'Sesudah'}, inplace=True)
st.dataframe(stemming_display.head(10))