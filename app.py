import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Klasifikasi Kemiskinan - KMeans", layout="centered")

st.title("📉 Prediksi Klaster Kemiskinan Berdasarkan P0")

st.markdown("Masukkan persentase penduduk miskin untuk melihat hasil klasifikasi:")

# Input tunggal
p0 = st.number_input("Persentase Penduduk Miskin (P0) [%]", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f")

if st.button("🔍 Prediksi Klaster"):
    # Data dummy (training KMeans hanya dengan P0)
    X_train = np.array([[5], [7], [8], [15], [17], [18]])
    y_dummy = [0, 0, 0, 1, 1, 1]

    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans.fit(X_train)

    # Prediksi
    user_data = np.array([[p0]])
    cluster = kmeans.predict(user_data)[0]

    st.subheader("📌 Hasil Prediksi")
    st.write(f"📍 Data Anda diklasifikasikan ke dalam **Klaster {cluster}**")

    # Visualisasi
    df_plot = pd.DataFrame(X_train, columns=["P0"])
    df_plot["Cluster"] = kmeans.labels_

    fig, ax = plt.subplots()
    sns.stripplot(data=df_plot, x="P0", hue="Cluster", palette="Set2", size=12, ax=ax)
    plt.scatter(p0, 0, color="red", s=150, marker="X", label="Input Anda")
    plt.legend()
    st.pyplot(fig)
