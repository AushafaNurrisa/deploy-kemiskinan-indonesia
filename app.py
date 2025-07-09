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

    # Bersihkan nama kolom
    df.columns = df.columns.str.strip()

    # Tangani koma desimal dan konversi ke float jika bisa
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.')
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    # Tampilkan data awal
    st.subheader("📊 Dataframe Awal")
    st.dataframe(df)

    # Tangani kolom klasifikasi
    if 'Klasifikasi Kemiskinan' in df.columns:
        try:
            df['Klasifikasi Kemiskinan'] = df['Klasifikasi Kemiskinan'].astype(float).astype(int)
        except:
            st.warning("Nilai pada kolom 'Klasifikasi Kemiskinan' tidak bisa dikonversi ke integer.")
    else:
        st.warning("Kolom 'Klasifikasi Kemiskinan' tidak ditemukan.")
        st.stop()

    # Visualisasi distribusi klasifikasi
    st.subheader("Distribusi Klasifikasi Kemiskinan")
    fig1, ax1 = plt.subplots()
    df.groupby('Klasifikasi Kemiskinan').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax1)
    ax1.set_title("Jumlah per Klasifikasi")
    ax1.set_xlabel("Jumlah")
    ax1.set_ylabel("Klasifikasi")
    st.pyplot(fig1)

    # Visualisasi histogram
    st.subheader("Histogram Persentase Penduduk Miskin")
    fig2, ax2 = plt.subplots()
    df['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'].plot(kind='hist', bins=20, ax=ax2)
    ax2.set_title("Distribusi P0")
    st.pyplot(fig2)

    # K-Means Clustering
    st.subheader("🔍 K-Means Clustering")
    X_cluster = df[['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)', 
                    'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)']]
    kmeans = KMeans(n_clusters=2, random_state=21)
    df['Klaster K-Means'] = kmeans.fit_predict(X_cluster)

    # Visualisasi hasil clustering
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x=X_cluster.columns[0], y=X_cluster.columns[1], hue='Klasifikasi Kemiskinan', ax=ax3)
    ax3.set_title("Data Asli")

    sns.scatterplot(data=df, x=X_cluster.columns[0], y=X_cluster.columns[1], hue='Klaster K-Means', palette='rainbow', ax=ax4)
    ax4.set_title("Hasil Klasterisasi")

    st.pyplot(fig3)

    # Akurasi (jika mungkin)
    try:
        acc = accuracy_score(df['Klasifikasi Kemiskinan'], df['Klaster K-Means'])
        st.success(f"Akurasi K-Means: {acc:.2f}")
    except:
        st.warning("Tidak bisa menghitung akurasi karena label tidak cocok.")
