#1 Import Library
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import plotly.graph_objects as go

url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

#data = conn.read(worksheet="Sheet1")
data = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3])

#3 Pre-Processing Data
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('Datetime', inplace=True)
data = data[['Index']]
data = data.resample('2T').mean()
data['Index'].interpolate(method='linear', inplace=True)

#4 Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
data ['Index_scaled'] = scaler.fit_transform(data[['Index']])

#5 Inisialisasi Timestep
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)-n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 7
X, y = prepare_data(data['Index_scaled'].values, n_steps)

#6 Split Data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#7 Bangun LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])

#8 Pembuatan Model dan Kompilasi Model
model.compile(optimizer='adam', loss='mean_squared_error')

#9 Pelatihan Model
history=model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

#10 Prediksi Model
train_predicted = model.predict(X_train)
test_predicted = model.predict(X_test)

last_time = data.index[-1]
last_time = last_time.replace(second=0, microsecond=0)
minute_offset = last_time.minute % 30
if minute_offset != 0:
    last_time += pd.Timedelta(minutes=(30 - minute_offset))  # Bulatkan ke atas

# Interval waktu 30 menit
time_interval = pd.Timedelta(minutes=30)

# Prediksi ke depan
future_steps = 10
last_sequence = X_test[-1]  # Ambil urutan terakhir dari data test
future_predictions = []
future_times = [last_time + i * time_interval for i in range(1, future_steps + 1)]

# Loop untuk prediksi
for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, n_steps, 1))[0, 0]
    future_predictions.append(prediction)
    last_sequence = np.append(last_sequence[1:], prediction)

# Inversi normalisasi dan bulatkan prediksi
future_predictions_scaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_df = pd.DataFrame({
    'Time': future_times,
    'Predicted Index': np.floor(future_predictions_scaled.flatten()).astype(int)
})

future_df = future_df[(future_df['Time'].dt.hour >= 6) & (future_df['Time'].dt.hour <= 18)]

# Custom Header
st.markdown(
    """
    <style>
    .header {
        background-color: #D6D6F5;
        padding: 20px; /* Tambahkan padding lebih untuk keseimbangan */
        text-align: center;
        border-radius: 5px;
        display: flex;
        flex-direction: column;
        align-items: center; /* Pusatkan elemen secara vertikal */
    }
    .header img {
        width: 80px; /* Sesuaikan ukuran gambar */
        margin-bottom: 10px; /* Tambahkan jarak di bawah logo */
    }
    </style>
    <div class="header">
        <img src="https://images.app.goo.gl/aAEAaxFEqJCiDQkRA" alt="Logo">
        <h2>Real-Time Sensor Data</h2>
        <p>(THE DISPLAYED DATA IS REAL-TIME)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit Title
st.title("UV Index")
latest_data = data.iloc[-1]  # Data terbaru
latest_time = latest_data.name  # Waktu dari indeks
uv_index = latest_data['Index']  # Nilai Index

# Membuat gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=uv_index,
    title={'text': "UV Level"},
    gauge={
        'axis': {'range': [0, 11]},
        'bar': {'color': "orange"},
        'steps': [
            {'range': [0, 3], 'color': "green"},
            {'range': [3, 6], 'color': "yellow"},
            {'range': [6, 8], 'color': "orange"},
            {'range': [8, 10], 'color': "red"},
            {'range': [10,11], 'color': "purple"},
        ]
    }
))

# Menampilkan widget
st.plotly_chart(fig, use_container_width=True)
st.write(f"**Time:** {latest_time.strftime('%H:%M')}")

st.title("UV Index Prediction")
# Tampilan grid prakiraan
cols = st.columns(len(future_df))
for i, row in future_df.iterrows():
    with cols[i]:
        uv_level = row["Predicted Index"]
        if uv_level < 3:
            icon = "🟢"
            desc = "Low"
            bg_color = "#00ff00"
        elif uv_level < 6:
            icon = "🟡"
            desc = "Moderate"
            bg_color = "#ffcc00"
        elif uv_level < 8:
            icon = "🟠"
            desc = "High"
            bg_color = "#ff6600"
        elif uv_level < 11:
            icon = "🔴"
            desc = "Very High"
            bg_color = "#ff0000"
        else:
            icon = "🟣"
            desc = "Extreme"
            bg_color = "#9900cc"
        
        # Kustomisasi tampilan grid
        st.markdown(
            f"""
            <div style="text-align:center; padding:10px; border-radius:5px; background-color:{bg_color};">
                <h3 style="color:white;">{row['Time'].strftime('%H:%M')}</h3>
                <h2 style="color:white;">{icon} {uv_level}</h2>
                <p style="color:white;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
# Custom Footer
st.markdown(
    """
    <style>
    .footer {
        background-color: #D6D6F5;
        padding: 20px; /* Padding yang sama seperti header */
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        border-radius: 5px; /* Tambahkan border-radius agar simetri */
        display: flex;
        flex-direction: column;
        align-items: center; /* Pusatkan elemen secara vertikal */
    }
    .footer img {
        width: 80px; /* Sesuaikan ukuran gambar dengan header */
        margin-bottom: 10px; /* Jarak di bawah logo */
    }
    </style>
    <div class="footer">
        <img src="https://images.app.goo.gl/aAEAaxFEqJCiDQkRA" alt="Logo">
        <p>Diponegoro University<br>Fakultas Sains dan Matematika<br>Departemen Fisika</p>
        <p>Nastangini<br>20440102130112</p>
    </div>
    """,
    unsafe_allow_html=True
)
