import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from streamlit_gsheets import GSheetsConnection
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Streamlit Title
st.title("UV Index Prediction using LSTM")

# Streamlit GSheets Connection
st.subheader("Load Data from Google Sheets")
url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)
data = conn.read(spreadsheet=url)
st.dataframe(data)

# Processing Data
st.subheader("Preprocessing Data")
# Convert Date and Time to Datetime
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('Datetime', inplace=True)
data = data[['Index']].copy()

# Resample Data and Interpolate Missing Values
data = data.resample('2T').mean()
data['Index'].interpolate(method='linear', inplace=True)

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Index_scaled'] = scaler.fit_transform(data[['Index']])

# Visualize Raw Data
st.line_chart(data['Index'])

# Prepare Data for LSTM
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)-n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 7
X, y = prepare_data(data['Index_scaled'].values, n_steps)

# Split Data into Train and Test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape Data for LSTM Input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
st.subheader("Training the Model")
with st.spinner("Training LSTM Model..."):
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predictions
train_predicted = model.predict(X_train)
test_predicted = model.predict(X_test)

# Evaluate Model
mse_train = mean_squared_error(y_train, train_predicted)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_predicted)
r2_train = r2_score(y_train, train_predicted) * 100

mse_test = mean_squared_error(y_test, test_predicted)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_predicted)
r2_test = r2_score(y_test, test_predicted) * 100

# Show Evaluation Metrics
st.subheader("Model Evaluation")
st.write("### Training Metrics")
st.write(f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.2f}%")
st.write("### Testing Metrics")
st.write(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.2f}%")

# Fungsi untuk melakukan prediksi masa depan
def predict_future(model, scaler, X_test, n_steps, future_steps, time_interval, start_time):
    last_sequence = X_test[-1]  # Ambil urutan terakhir dari data test
    future_predictions = []
    future_times = [start_time + i * time_interval for i in range(1, future_steps + 1)]

    # Loop untuk prediksi
    for _ in range(future_steps):
        prediction = model.predict(last_sequence.reshape(1, n_steps, 1))[0, 0]
        future_predictions.append(prediction)
        last_sequence = np.append(last_sequence[1:], prediction)  # Perbarui urutan

    # Inversi normalisasi dan pembulatan ke integer
    future_predictions_scaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_df = pd.DataFrame({
        'Time': future_times,
        'Predicted Index': np.floor(future_predictions_scaled.flatten()).astype(int)
    })

    # Filter prediksi hanya untuk jam antara 06:00 dan 18:00
    future_df = future_df[(future_df['Time'].dt.hour >= 6) & (future_df['Time'].dt.hour <= 18)]
    return future_df

# Streamlit Title
st.title("Future UV Index Prediction")

# Sidebar untuk konfigurasi prediksi
start_time = st.sidebar.time_input("Start Time", value=pd.Timestamp.now().replace(second=0, microsecond=0))
future_steps = st.sidebar.slider("Number of Future Steps", min_value=1, max_value=20, value=10)
time_interval = pd.Timedelta(minutes=30)

# Prediksi masa depan jika tombol ditekan
if st.sidebar.button("Predict Future"):
    # Pastikan model, scaler, X_test, dan n_steps sudah didefinisikan sebelumnya
    future_df = predict_future(model, scaler, X_test, n_steps, future_steps, time_interval, start_time)

    # Tampilkan DataFrame hasil prediksi
    st.subheader("Future Predictions")
    st.dataframe(future_df)

    # Visualisasi prediksi
    st.subheader("Prediction Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(future_df['Time'], future_df['Predicted Index'], marker='o', label='Future Predictions', color='purple')
    ax.set_title('Future UV Index Predictions', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('UV Index', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Format sumbu waktu
    ax.set_xticks(future_df['Time'])
    ax.set_xticklabels(future_df['Time'].dt.strftime('%H:%M'), rotation=45)
    st.pyplot(fig)
