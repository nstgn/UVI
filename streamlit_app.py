# Import Library
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# Streamlit Title
st.title("Prediksi Indeks UV Menggunakan LSTM")

# Konfigurasi Kredensial Google Sheets
st.subheader("Mengakses Data dari Google Sheets")
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    client = gspread.authorize(creds)
    sheet_url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit"
    worksheet = client.open_by_url(sheet_url).sheet1
    data = pd.DataFrame(worksheet.get_all_records())
    st.success("Data berhasil diakses!")
except Exception as e:
    st.error(f"Error: {e}")

# Visualisasi Data Mentah
st.subheader("Visualisasi Data Mentah")
plt.figure(figsize=(10, 4))
plt.plot(data['Index'], label='Raw Index Data', color='purple')
plt.title("Raw UV Index Data")
plt.xlabel("Time Steps")
plt.ylabel("UV Index")
plt.legend()
st.pyplot(plt)

# Pre-Processing Data
st.subheader("Preprocessing Data")
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('Datetime', inplace=True)
data = data[['Index']].copy()
data = data.resample('2T').mean()
data['Index'].interpolate(method='linear', inplace=True)

# Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Index_scaled'] = scaler.fit_transform(data[['Index']])

# Inisialisasi Timestep
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 7
X, y = prepare_data(data['Index_scaled'].values, n_steps)

# Split Data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Bangun Model LSTM
st.subheader("Membangun Model LSTM")
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Latih Model
st.subheader("Training Model")
if st.button("Mulai Training"):
    with st.spinner("Training model, mohon tunggu..."):
        history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    st.success("Training selesai!")

# Evaluasi Model
st.subheader("Evaluasi Model")
train_predicted = model.predict(X_train)
test_predicted = model.predict(X_test)
mse_train = mean_squared_error(y_train, train_predicted)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_predicted)
r2_train = r2_score(y_train, train_predicted) * 100
mse_test = mean_squared_error(y_test, test_predicted)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_predicted)
r2_test = r2_score(y_test, test_predicted) * 100
st.write(f"Training Metrics: MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.2f}%")
st.write(f"Testing Metrics: MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.2f}%")

# Prediksi Masa Depan
st.subheader("Prediksi Masa Depan")
future_steps = 10
time_interval = pd.Timedelta(minutes=30)
last_sequence = X_test[-1]
future_predictions = []
last_time = data.index[-1]
for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, n_steps, 1))[0, 0]
    future_predictions.append(prediction)
    last_sequence = np.append(last_sequence[1:], prediction)
future_predictions_scaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_times = [last_time + i * time_interval for i in range(1, future_steps + 1)]
future_df = pd.DataFrame({'Time': future_times, 'Predicted Index': np.floor(future_predictions_scaled.flatten()).astype(int)})
st.write(future_df)

# Visualisasi Prediksi Masa Depan
plt.figure(figsize=(10, 6))
plt.plot(future_df['Time'], future_df['Predicted Index'], marker='o', label='Future Predictions', color='purple')
plt.title('Future UV Index Predictions')
plt.xlabel('Time')
plt.ylabel('UV Index')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(plt)
