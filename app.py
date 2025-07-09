import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Klasifikasi Kemiskinan - K-Means", layout="centered")

st.title("📉 Klasifikasi Tingkat Kemiskinan di Indonesia dengan K-Means")
st.markdown("Masukkan dataset berikut untuk melakukan klasifikasi kemiskinan berdasarkan variabel sosial ekonomi:")

# Form upload
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Preprocessing
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.')
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    if 'Klasifikasi Kemiskinan' not in df.columns:
        st.error("❌ Kolom 'Klasifikasi Kemiskinan' tidak ditemukan.")
        st.stop()

    try:
        df['Klasifikasi Kemiskinan'] = df['Klasifikasi Kemiskinan'].astype(float).astype(int)
    except:
        st.warning("⚠️ Gagal mengonversi 'Klasifikasi Kemiskinan' ke integer")

    # Visualisasi
    st.subheader("📊 Distribusi Klasifikasi")
    fig1, ax1 = plt.subplots()
    df['Klasifikasi Kemiskinan'].value_counts().sort_index().plot(kind='barh', color='skyblue', ax=ax1)
    ax1.set_xlabel("Jumlah")
    ax1.set_ylabel("Klasifikasi")
    ax1.set_title("Jumlah Data per Klasifikasi")
    st.pyplot(fig1)

    # Clustering
    st.subheader("🔍 K-Means Clustering")
    kolom_x = 'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'
    kolom_y = 'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)'

    if kolom_x not in df.columns or kolom_y not in df.columns:
        st.error(f"Kolom '{kolom_x}' dan/atau '{kolom_y}' tidak ditemukan.")
        st.stop()

    try:
        X_cluster = df[[kolom_x, kolom_y]].dropna()
        kmeans = KMeans(n_clusters=2, random_state=21, n_init='auto')
        df['Klaster K-Means'] = kmeans.fit_predict(X_cluster)
    except Exception as e:
        st.error(f"Gagal menjalankan K-Means: {e}")
        st.stop()

    # Visualisasi hasil klaster
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(data=df, x=kolom_x, y=kolom_y, hue='Klasifikasi Kemiskinan', ax=ax2)
    ax2.set_title("Data Asli")

    sns.scatterplot(data=df, x=kolom_x, y=kolom_y, hue='Klaster K-Means', palette='rainbow', ax=ax3)
    ax3.set_title("Hasil K-Means")

    st.pyplot(fig2)

    # Akurasi
    try:
        y_true = df['Klasifikasi Kemiskinan']
        y_pred = df['Klaster K-Means']
        acc = max(
            accuracy_score(y_true, y_pred),
            accuracy_score(y_true, 1 - y_pred)
        )
        st.success(f"✅ Akurasi K-Means: {acc:.2f}")
    except:
        st.warning("Tidak bisa menghitung akurasi. Periksa kembali label dan hasil klaster.")
