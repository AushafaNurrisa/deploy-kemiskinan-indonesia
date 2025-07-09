import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Klasifikasi Kemiskinan - K-Means", layout="centered")

st.title("📉 Prediksi Klaster Kemiskinan Berdasarkan Data Sosial Ekonomi")

st.markdown("Masukkan data berikut untuk melihat hasil klasifikasi berdasarkan model K-Means:")

# Input fitur
p0 = st.number_input("Persentase Penduduk Miskin (P0) [%]", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f")
lama_sekolah = st.number_input("Rata-rata Lama Sekolah Penduduk 15+ [tahun]", min_value=0.0, max_value=20.0, value=7.0, step=0.1, format="%.2f")

# Tombol
if st.button("🔍 Prediksi Klaster"):
    # Data dummy untuk pelatihan model
    X_train = np.array([
        [5, 9], [7, 8], [8, 7],  # cluster 0
        [15, 5], [17, 4], [18, 3]  # cluster 1
    ])
    y_dummy = [0, 0, 0, 1, 1, 1]

    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans.fit(X_train)

    # Prediksi input user
    user_data = np.array([[p0, lama_sekolah]])
    cluster = kmeans.predict(user_data)[0]

    st.subheader("📌 Hasil Prediksi")
    st.write(f"📍 Data Anda diklasifikasikan ke dalam **Klaster {cluster}**")

    # Visualisasi cluster dummy + input
    df_plot = pd.DataFrame(X_train, columns=['P0', 'Lama Sekolah'])
    df_plot['Cluster'] = kmeans.labels_

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_plot, x='P0', y='Lama Sekolah', hue='Cluster', palette='Set2', s=100, ax=ax)
    plt.scatter(p0, lama_sekolah, color='red', label='Input Anda', s=150, marker='X')
    plt.legend()
    st.pyplot(fig)
