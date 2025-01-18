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

# Visualize Predictions
st.subheader("Predictions vs Actual")
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
train_predicted = scaler.inverse_transform(train_predicted)
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
test_predicted = scaler.inverse_transform(test_predicted)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(train_actual)), train_actual, label='Actual Train', color='blue')
ax.plot(range(len(train_actual), len(train_actual) + len(test_actual)), test_actual, label='Actual Test', color='green')
ax.plot(range(len(train_predicted)), train_predicted, label='Predicted Train', color='red')
ax.plot(range(len(train_predicted), len(train_predicted) + len(test_predicted)), test_predicted, label='Predicted Test', color='orange')
ax.set_title('Predictions vs Actual')
ax.set_xlabel('Time Steps')
ax.set_ylabel('UV Index')
ax.legend()
st.pyplot(fig)

# Future Predictions
st.subheader("Future Predictions")
future_steps = 10
last_sequence = X_test[-1]
future_predictions = []
for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, n_steps, 1))[0, 0]
    future_predictions.append(prediction)
    last_sequence = np.append(last_sequence[1:], prediction)

future_predictions_scaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_df = pd.DataFrame({
    'Step': range(1, future_steps + 1),
    'Predicted Index': future_predictions_scaled.flatten()
})

st.write(future_df)
st.line_chart(future_df.set_index('Step')['Predicted Index'])


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_model.h5")
    scaler = MinMaxScaler()
    return model, scaler

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

# Prediksi ke depan
n_steps = 10
last_sequence = data_scaled[-n_steps:]
future_steps = 10
time_interval = pd.Timedelta(minutes=30)
future_times = [data.index[-1] + (i + 1) * time_interval for i in range(future_steps)]
future_predictions = []

# Lakukan prediksi masa depan
for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, n_steps, 1))[0, 0]
    future_predictions.append(prediction)
    last_sequence = np.append(last_sequence[1:], prediction)

# Inversi normalisasi prediksi
future_predictions_scaled = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)
future_df = pd.DataFrame({
    "Time": future_times,
    "Predicted Index": np.floor(future_predictions_scaled.flatten()).astype(int)
})

# Filter waktu (06:00 - 18:00 saja)
future_df = future_df[(future_df["Time"].dt.hour >= 6) & (future_df["Time"].dt.hour <= 18)]

# Tampilkan prakiraan UV di Streamlit
st.title("UV Index Forecast")
st.write("Prakiraan UV Index untuk 5 Jam ke Depan")

# Tampilan grid prakiraan
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

# Visualisasi
st.write("---")
st.subheader("Visualisasi UV Index Prediksi")
plt.figure(figsize=(10, 6))
plt.plot(future_df["Time"], future_df["Predicted Index"], marker="o", color="purple", label="Predicted UV Index")
plt.title("Predicted UV Index")
plt.xlabel("Time")
plt.ylabel("UV Index")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
st.pyplot(plt)
