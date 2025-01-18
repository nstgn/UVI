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
import matplotlib.pyplot as plt

# Streamlit Title
st.title("UV Index Prediction using LSTM")

url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

#data = conn.read(worksheet="Sheet1")
data = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3])

#3 Pre-Processing Data
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('Datetime', inplace=True)
data = data[['Index']].copy()

# Resampling data setiap 2 menit, interpolasi nilai yang hilang
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

#11 Evaluasi Model
#Evaluasi untuk data training
mse_train = mean_squared_error(y_train,train_predicted)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_predicted)
r2_train = r2_score(y_train,train_predicted)*100

#Evaluasi untuk data testing
mse_test = mean_squared_error(y_test, test_predicted)
rmse_test =np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_predicted)
r2_test = r2_score(y_test, test_predicted)*100

# Menampilkan hasil evaluasi
st.subheader("Model Evaluation")
st.write("### Training Metrics")
st.write(f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.2f}%")
st.write("### Testing Metrics")
st.write(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.2f}%")


#13 Visualisasi Data
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

# Atur waktu awal ke interval 30 menit terdekat
st.subheader("Future Predictions")
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
