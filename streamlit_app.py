import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import timedelta

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_model.h5")
    scaler = MinMaxScaler()
    return model, scaler

# Fungsi untuk prediksi masa depan
def predict_future(model, data_scaled, n_steps, scaler, future_steps=10):
    last_sequence = data_scaled[-n_steps:]
    future_predictions = []
    for _ in range(future_steps):
        prediction = model.predict(last_sequence.reshape(1, n_steps, 1))[0, 0]
        future_predictions.append(prediction)
        last_sequence = np.append(last_sequence[1:], prediction)

    future_predictions_scaled = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )
    return future_predictions_scaled.flatten()

# Fungsi untuk visualisasi
def plot_uv_index(future_df):
    plt.figure(figsize=(10, 6))
    plt.plot(
        future_df["Time"],
        future_df["Predicted Index"],
        marker="o",
        color="purple",
        label="Predicted UV Index",
    )
    plt.title("Predicted UV Index")
    plt.xlabel("Time")
    plt.ylabel("UV Index")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    st.pyplot(plt)

# Load model dan scaler
model, scaler = load_lstm_model()

# Data dummy: Anda bisa mengganti ini dengan data Anda
data = pd.DataFrame({
    "Time": pd.date_range(start="2025-01-01 06:00", periods=100, freq="30T"),
    "UV Index": np.random.rand(100) * 10
})
data.set_index("Time", inplace=True)

# Normalisasi data
data_scaled = scaler.fit_transform(data)

# Sidebar untuk input prediksi
st.sidebar.title("Prediction Settings")
n_steps = st.sidebar.slider("Steps for Prediction", min_value=1, max_value=20, value=10)
future_steps = st.sidebar.slider("Future Steps to Predict", min_value=1, max_value=20, value=10)

# Prediksi masa depan
future_predictions = predict_future(model, data_scaled, n_steps, scaler, future_steps)
time_interval = timedelta(minutes=30)
future_times = [data.index[-1] + (i + 1) * time_interval for i in range(future_steps)]

future_df = pd.DataFrame({
    "Time": future_times,
    "Predicted Index": np.floor(future_predictions).astype(int)
})

# Filter waktu (06:00 - 18:00 saja)
future_df = future_df[(future_df["Time"].dt.hour >= 6) & (future_df["Time"].dt.hour <= 18)]

# Tampilkan prakiraan UV
st.title("UV Index Forecast")
st.write("Prakiraan UV Index untuk 5 Jam ke Depan")
cols = st.columns(len(future_df))
for i, row in future_df.iterrows():
    with cols[i]:
        uv_level = row["Predicted Index"]
        if uv_level < 3:
            icon = "🟢"
            desc = "Low"
        elif uv_level < 6:
            icon = "🟡"
            desc = "Moderate"
        elif uv_level < 8:
            icon = "🟠"
            desc = "High"
        elif uv_level < 11:
            icon = "🔴"
            desc = "Very High"
        else:
            icon = "🟣"
            desc = "Extreme"
        st.markdown(f"### {row['Time'].strftime('%H:%M')}")
        st.markdown(f"#### {icon} {uv_level}")
        st.markdown(desc)

# Visualisasi UV Index
st.write("---")
st.subheader("Visualisasi UV Index Prediksi")
plot_uv_index(future_df)
