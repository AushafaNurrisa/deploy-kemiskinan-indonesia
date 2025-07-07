import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Judul Halaman
st.title("Klasifikasi Kemiskinan di Indonesia dengan K-Means")

# Upload file CSV
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Preprocessing
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.')
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    df = df.dropna(how='all')
    try:
        df['Klasifikasi Kemiskinan'] = df['Klasifikasi Kemiskinan'].astype(int)
    except:
        st.warning("Kolom klasifikasi tidak tersedia atau tidak valid")

    st.subheader("Dataframe")
    st.dataframe(df)

    # Visualisasi
    st.subheader("Distribusi Klasifikasi Kemiskinan")
    fig1, ax1 = plt.subplots()
    df.groupby('Klasifikasi Kemiskinan').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax1)
    st.pyplot(fig1)

    st.subheader("Histogram Persentase Penduduk Miskin")
    fig2, ax2 = plt.subplots()
    df['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'].plot(kind='hist', bins=20, ax=ax2)
    st.pyplot(fig2)

    # K-Means Clustering
    st.subheader("K-Means Clustering")
    X_cluster = df[['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)', 'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)']]
    kmeans = KMeans(n_clusters=2, random_state=21)
    df['Klaster K-Means'] = kmeans.fit_predict(X_cluster)

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x=X_cluster.columns[0], y=X_cluster.columns[1], hue='Klasifikasi Kemiskinan', ax=ax3)
    ax3.set_title("Data Asli")
    sns.scatterplot(data=df, x=X_cluster.columns[0], y=X_cluster.columns[1], hue='Klaster K-Means', palette='rainbow', ax=ax4)
    ax4.set_title("Hasil Klasterisasi")
    st.pyplot(fig3)

klasifikasi_cols = [col for col in df.columns if 'klasifikasi' in col.lower()]

if len(klasifikasi_cols) > 0:
    try:
        df[klasifikasi_cols[0]] = df[klasifikasi_cols[0]].astype(int)
        df.rename(columns={klasifikasi_cols[0]: 'Klasifikasi Kemiskinan'}, inplace=True)
    except:
        st.warning(f"Kolom ditemukan: {klasifikasi_cols[0]}, tapi tidak bisa dikonversi ke integer.")
else:
    st.warning("Kolom klasifikasi tidak tersedia atau tidak dikenali. Kolom yang tersedia:")
    st.write(df.columns.tolist())
