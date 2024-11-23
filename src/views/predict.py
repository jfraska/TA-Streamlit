import streamlit as st
import pandas as pd
import os
import re
import itertools
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../assets')

def noise_removal(text):
    # remove excessive newline
    text = re.sub('\n', ' ',text)

    # remove URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)

    # remove twitter & instagram formatting
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'\brt\b', '', text)

    # remove kaskus formatting
    text = re.sub('\[', ' [', text)
    text = re.sub('\]', '] ', text)
    text = re.sub('\[quote[^ ]*\].*?\[\/quote\]', ' ', text)
    text = re.sub('\[[^ ]*\]', ' ', text)
    text = re.sub('&quot;', ' ', text)

    # remove emojis
    emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    # remove non alphabet
    text = re.sub('[^a-zA-Z ]+', '', text)

    # remove repeating characters
    text = ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

    # remove excessive whitespace
    text = re.sub('  +', ' ', text)

    return text

def case_folding(text):
    return text.lower()

slang_words = pd.read_csv(os.path.join(file_path, 'slangword.csv'))
slang_dict = dict(zip(slang_words['original'],slang_words['translated']))

def transform_slang_words(text):
    word_list = text.split()
    word_list_len = len(word_list)
    transformed_word_list = []
    i = 0
    while i < word_list_len:
        if (i + 1) < word_list_len:
            two_words = ' '.join(word_list[i:i+2])
            if two_words in slang_dict:
                transformed_word_list.append(slang_dict[two_words])
                i += 2
                continue
        transformed_word_list.append(slang_dict.get(word_list[i], word_list[i]))
        i += 1
    return ' '.join(transformed_word_list)

stopword_factory = StopWordRemoverFactory().get_stop_words()
more_stopword = ['ada']

data = stopword_factory + more_stopword
dictionary = ArrayDictionary(data)

def stopword(text):
    stopword = StopWordRemover(dictionary)
    return stopword.remove(text)

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def stemming(text):
  return stemmer.stem(text)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained('indolem/indobert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.linear = torch.nn.Linear(768, NUM_LABELS)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output



# --- HERO SECTION ---
st.title('Klasifikasi Multi-Label Komentar')
input_text = st.text_area("Masukan komentar:", '"#SidangAhok smg sipenista agama n ateknya matinya tdk wjar n jasadnya tdk dtrma tnh n dia tdk prnh mrs kn sorga,aamiin semoga tuhan setuju"')

tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Hyperparameters
MAX_LEN = 256
NUM_LABELS = 4
DROPOUT = 0.4

# Loading fine-tunning model (best model)
model = BERTClass()
model.load_state_dict(torch.load(os.path.join(file_path,"MLTC_model_state.bin"),map_location=torch.device('cpu')))
model = model.to(device)

target_list = ['Pornografi', 'SARA', 'Radikalisme', 'Pencemaran Nama Baik']

if st.button('Klasifikasi'):
    
    # Noise Removal
    input_text_noise = noise_removal(input_text)
    st.subheader("1. Noise Removal")
    st.write(input_text_noise)

    # Case Folding
    input_text_lower = case_folding(input_text_noise)
    st.subheader("2. Case Folding")
    st.write(input_text_lower)

    # Transform Slang Words
    input_text_slang = transform_slang_words(input_text_lower)
    st.subheader("3. Translate Slang Words")
    st.write(input_text_slang)

    # Stopword Removal
    input_text_stopwords = stopword(input_text_slang)
    st.subheader("4. Hapus Stopwords")
    st.write(input_text_stopwords)

    # Stemming
    input_text_stemmed = stemming(input_text_stopwords)
    st.subheader("5. Stemming")
    st.write(input_text_stemmed)

    # Tambahkan spasi untuk tampilan yang lebih bersih
    st.markdown("---")  # Garis pemisah

    encoded_text = tokenizer.encode_plus(
    input_text_stemmed,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
    )

    tokens = tokenizer.tokenize(input_text_stemmed,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt')
    
    # Menampilkan hasil tokenisasi
    st.subheader("BERT Tokenisasi :")
    
    tokens_df = pd.DataFrame(tokens)
    tokens_df.columns = ['token']
    st.write(tokens_df)

    # st.write("Input IDs:", encoded_text['input_ids'].numpy())

    # Tambahkan spasi untuk tampilan yang lebih bersih
    st.markdown("---")  # Garis pemisah

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    token_type_ids = encoded_text['token_type_ids'].to(device)
    
    output = model(input_ids, attention_mask, token_type_ids)
    # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
    output = torch.sigmoid(output).detach().cpu()
    
    # Menampilkan hasil sigmoid
    st.subheader("Hasil Sigmoid :")
    sigmoid = output.squeeze()
    df_sigmoid = pd.DataFrame(sigmoid.numpy(), index=target_list)
    df_sigmoid.columns = ['probabilitas']
    st.write(df_sigmoid)
    
    # thresholding at 0.5
    output = output.flatten().round().numpy()
        
    df = pd.DataFrame(output, index=target_list)
    df.columns = ['probabilitas']
    st.write(df)
        
    # Tambahkan spasi untuk tampilan yang lebih bersih
    st.markdown("---")  # Garis pemisah    
        
     # Display the result
    st.subheader("Hasil Klasifikasi:")
    
    label_found = False 
    for idx, p in enumerate(output):
        if p == 1:
            label_found = True
            # Tampilkan label dalam format yang lebih menarik
            st.markdown(f"<div style='padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; background-color: #E8F5E9; color: black;'>"
                        f"<strong>Label:</strong> {target_list[idx]}</div>", unsafe_allow_html=True)
    
    if not label_found:
        st.markdown("<div style='padding: 10px; border: 1px solid #F44336; border-radius: 5px; background-color: #FFEBEE; color: black;'>"
                    "<strong>Label:</strong> Tidak ada label ditemukan</div>", unsafe_allow_html=True)