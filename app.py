import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Klasifikasi Kemiskinan di Indonesia", layout="wide")

st.title("Klasifikasi Kemiskinan di Indonesia dengan K-Means")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Bersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()

    # Tangani koma desimal
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.')
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    # Tampilkan data
    st.subheader("📊 Dataframe Awal")
    st.dataframe(df)

    # Validasi kolom klasifikasi
    if 'Klasifikasi Kemiskinan' not in df.columns:
        st.error("Kolom 'Klasifikasi Kemiskinan' tidak ditemukan di file CSV.")
        st.stop()

    try:
        df['Klasifikasi Kemiskinan'] = df['Klasifikasi Kemiskinan'].astype(float).astype(int)
    except:
        st.warning("Kolom 'Klasifikasi Kemiskinan' tidak bisa dikonversi ke integer.")

    # Visualisasi distribusi klasifikasi
    st.subheader("Distribusi Klasifikasi Kemiskinan")
    fig1, ax1 = plt.subplots()
    df.groupby('Klasifikasi Kemiskinan').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax1)
    ax1.set_title("Jumlah per Klasifikasi")
    ax1.set_xlabel("Jumlah")
    ax1.set_ylabel("Klasifikasi")
    st.pyplot(fig1)

    # Validasi kolom yang dibutuhkan untuk clustering
    kolom_x = 'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'
    kolom_y = 'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)'

    if kolom_x not in df.columns or kolom_y not in df.columns:
        st.error(f"Kolom '{kolom_x}' dan/atau '{kolom_y}' tidak ditemukan.")
        st.stop()

    # Clustering
    st.subheader("🔍 K-Means Clustering")
    try:
        X_cluster = df[[kolom_x, kolom_y]].dropna()
        kmeans = KMeans(n_clusters=2, random_state=21, n_init='auto')
        df['Klaster K-Means'] = kmeans.fit_predict(X_cluster)
    except Exception as e:
        st.error(f"Gagal menjalankan K-Means: {e}")
        st.stop()

    # Visualisasi hasil clustering
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df, x=kolom_x, y=kolom_y, hue='Klasifikasi Kemiskinan', ax=ax3)
    ax3.set_title("Data Asli")

    sns.scatterplot(data=df, x=kolom_x, y=kolom_y, hue='Klaster K-Means', palette='rainbow', ax=ax4)
    ax4.set_title("Hasil Klasterisasi")

    st.pyplot(fig3)

    # Akurasi
    try:
        y_true = df['Klasifikasi Kemiskinan']
        y_pred = df['Klaster K-Means']
        acc = max(
            accuracy_score(y_true, y_pred),
            accuracy_score(y_true, 1 - y_pred)  # Antisipasi kebalikan label
        )
        st.success(f"Akurasi K-Means: {acc:.2f}")
    except:
        st.warning("Tidak dapat menghitung akurasi. Periksa kembali label dan hasil klaster.")
