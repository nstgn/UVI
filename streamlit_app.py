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

url = "https://docs.google.com/spreadsheets/d/1XPGPw08xkov3ly_BgamdInZUcSONPGbDSm0pOJ2ZJbw/edit?usp=sharing"
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
    last_time += pd.Timedelta(minutes=(30 - minute_offset))

# Interval waktu 30 menit
time_interval = pd.Timedelta(minutes=30)

# Prediksi ke depan
future_steps = 10
last_sequence = X_test[-1]
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
        padding: 10px;
        text-align: center;
        border-radius: 7px;
    }
    .header img {
        width: 60px;
    }
    </style>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/id/2/2d/Undip.png" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit Title
st.markdown(
    """
    <h1 style="text-align: center; ">UV Index</h1>
    """,
    unsafe_allow_html=True,
)

latest_data = data.iloc[-1] 
latest_time = latest_data.name 
uv_index = latest_data['Index'] 

# Membuat gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=uv_index,
    gauge={
        'axis': {'range': [0, 11]},
        'bar': {'color': "#3098ff"},
        'steps': [
            {'range': [0, 3], 'color': "#00ff00"},
            {'range': [3, 6], 'color': "#ffff00"},
            {'range': [6, 8], 'color': "#ff6600"},
            {'range': [8, 10], 'color': "#ff0000"},
            {'range': [10,11], 'color': "#9900cc"},
        ]
    }
))

# Menampilkan widget
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    f"""
    <div style="text-align: center;">
        <span style="display: inline-block; padding: 5px 15px; border-radius: 5px;
                    background-color: {'#d4edda' if uv_index <= 2 else '#fcfac0' if uv_index <= 5 else '#ffc78f' if uv_index <= 7 else '#ff8a8a' if uv_index <= 10 else '#e7cafc'};">
            {"<p style='color: #00ff00;'><strong>✅ Tingkat aman:</strong> Gunakan pelembab tabir surya SPF 30+ dan kacamata hitam.</p>" if uv_index <= 2 else
             "<p style='color: #ffcc00;'><strong>⚠️ Tingkat bahaya sedang:</strong> Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam, kenakan pakaian pelindung matahari.</p>" if uv_index <= 5 else
             "<p style='color: #ff6600;'><strong>⚠️ Tingkat bahaya tinggi:</strong> Kurangi paparan matahari antara pukul 10 pagi hingga pukul 4 sore.</p>" if uv_index <= 7 else
             "<p style='color: #ff0000;'><strong>⚠️ Tingkat bahaya sangat tinggi:</strong> Tetap di tempat teduh dan oleskan sunscreen setiap 2 jam.</p>" if uv_index <= 10 else
             "<p style='color: #9900cc;'><strong>❗ Tingkat bahaya ekstrem:</strong> Diperlukan semua tindakan pencegahan karena kulit dan mata dapat rusak dalam hitungan menit.</p>"}
       </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Menambahkan widget waktu
st.markdown(
    f"""
    <div style="text-align: center; font-size: medium; margin-top: 10px; margin-bottom: 40px;">
        <p><b>Pukul:</b> {latest_time.strftime('%H:%M')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="text-align: center;">UV Index Prediction</h1>
    """,
    unsafe_allow_html=True,
)

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

# Menambahkan tabel saran proteksi
st.markdown(
    """
    <h1 style="text-align: center;margin-top: 40px; margin-bottom: 10px;">Tabel Saran Proteksi</h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <table style="width:100%; border-collapse: collapse; text-align: center;">
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Kategori</th>
            <th style="border: 1px solid black; padding: 8px;">Himbauan</th>
        </tr>
        <tr style="background-color: #00ff00;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">0-2 (Low)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya rendah bagi orang banyak.</li>
                    <li>Kenakan kacamata hitam pada hari yang cerah.</li>
                    <li>Gunakan cairan pelembab tabir surya SPF 30+ bagi kulit sensitif.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #ffff00;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">3-5 (Moderate)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya sedang bagi orang yang terpapar matahari tanpa pelindung.</li>
                    <li>Tetap di tempat teduh pada saat matahari terik siang hari.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #ff6600;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">6-7 (High)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya tinggi bagi orang yang terpapar matahari tanpa pelindung, diperlukan pelindung untuk menghindari kerusakan mata dan kulit.</li>
                    <li>Kurangi waktu di bawah paparan matahari antara pukul 10 pagi hingga pukul 4 sore.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #ff0000;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">8-10 (Very High)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya tinggi bagi orang yang terpapar matahari tanpa pelindung, diperlukan pelindung untuk menghindari kerusakan mata dan kulit.</li>
                    <li>Minimal waktu di bawah paparan matahari antara pukul 10 pagi hingga pukul 4 sore.</li>
                    <li>Tetap di tempat teduh pada saat matahari terik siang hari.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #9900cc;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">11+ (Extreme)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya ekstrem, diperlukan semua tindakan pencegahan karena kulit dan mata dapat rusak dalam hitungan menit.</li>
                    <li>Hindari paparan matahari langsung dan pastikan perlindungan maksimal.</li>
                    <li>Tetap di tempat teduh pada saat matahari terik siang hari.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True,
)

# Custom Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 70px;
        font-size: 12px;
        text-align: left;
        margin: 0;
        padding: 5px 10px;
    }
    </style>
    <div class="footer">
        <p>Universitas Diponegoro<br>Fakultas Sains dan Matematika<br>Departemen Fisika</p>
        <p>Nastangini<br>20440102130112</p>
    </div>
    """,
    unsafe_allow_html=True
)
