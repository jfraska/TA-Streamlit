import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def show_confusion_matrix(confusion_matrix, label_name):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.title(f'Confusion Matrix for Label: {label_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)
    plt.clf()

def load_predictions_from_pickle(file_path):
    # Membuka file pickle dan memuat data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Mendapatkan data dari dictionary
    titles = data['titles']
    predictions = data['predictions']
    prediction_probs = data['prediction_probs']
    target_values = data['target_values']
    test_acc = data['test_acc']
    test_loss = data['test_loss']
    history = data['history']

    return titles, predictions, prediction_probs, target_values, test_acc, test_loss, history

# --- HERO SECTION ---
st.title('Laporan Hasil Training')

target_list = ['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']

# --- LOAD DATA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../assets')

titles, predictions, prediction_probs, target_values, test_acc, test_loss, history = load_predictions_from_pickle(os.path.join(file_path,"predictions.pkl"))

multilabel_cm = multilabel_confusion_matrix(target_values, predictions)
 
# Tabs for classification report and confusion matrix
tabs = st.tabs(['Reports', 'Confusion Matrix'])

with tabs[0]:
    # Menampilkan test accuracy dan test loss
    st.write(f"**Accuracy**: {test_acc:.4f}")
    st.write(f"**Loss**: {test_loss:.4f}")

    # Tambahkan spasi untuk tampilan yang lebih bersih
    st.markdown("---")  # Garis pemisah

    plt.rcParams["figure.figsize"] = (10,7)
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.grid()
    st.pyplot(plt)

    plt.clf()
    
    
    plt.rcParams["figure.figsize"] = (10,7)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.grid()    
    st.pyplot(plt)

    plt.clf()
    
    # Tambahkan spasi untuk tampilan yang lebih bersih
    st.markdown("---")  # Garis pemisah


    # Menghasilkan laporan klasifikasi dalam bentuk dictionary
    report_dict = classification_report(target_values, predictions, target_names=target_list, output_dict=True)

    # Konversi ke DataFrame untuk ditampilkan dalam format tabel
    report_df = pd.DataFrame(report_dict).transpose()

    # Tampilkan dalam bentuk tabel
    st.table(report_df)
    
with tabs[1]:
    
    for idx, cm in enumerate(multilabel_cm):
        df_cm = pd.DataFrame(cm)
        show_confusion_matrix(df_cm,  target_list[idx])
    
