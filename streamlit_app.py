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


url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

#data = conn.read(worksheet="Sheet1")
data = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3])
st.dataframe(data)

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


# Visualisasi Data Train dan Test

plt.plot(range(len(y_train)), y_train, label='Data Train', color='purple')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Data Test', color='green')
plt.title("Data Train dan Data Test")
plt.xlabel("Index") #karna default untuk urutan aja, jadi ya ga perlu pake
plt.ylabel("Scaled Intensity")
plt.legend()

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
print("Training Metrics:")
print(f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.2f}%")
print("\nTesting Metrics:")
print(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.2f}%")


#13 Visualisasi Data
import seaborn as sns
sns.set_style("darkgrid")
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
train_predicted = scaler.inverse_transform(train_predicted)
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
test_predicted = scaler.inverse_transform(test_predicted)

plt.figure(figsize=(5, 3))
plt.plot(range(len(train_actual)), train_actual, label='Actual Train', color='blue')
plt.plot(range(len(train_actual), len(train_actual) + len(test_actual)), test_actual, label='Actual Test', color='green')
plt.plot(range(len(train_predicted)), train_predicted, label='Predicted Train', color='red')
plt.plot(range(len(train_predicted), len(train_predicted) + len(test_predicted)), test_predicted, label='Predicted Test', color='orange')
plt.title('Prediksi vs Aktual (Latih dan Uji)')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.show()

# Atur waktu awal ke interval 30 menit terdekat
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

# Tampilkan hasil prediksi
print(future_df)

# Visualisasi prediksi ke depan
plt.figure(figsize=(10, 6))
plt.plot(future_df['Time'], future_df['Predicted Index'], marker='o', label='Future Predictions', color='purple')
plt.title('Future UV Index Predictions', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('UV Index', fontsize=12)

# Atur tampilan waktu pada sumbu X agar lebih rapi
plt.xticks(future_df['Time'], labels=future_df['Time'].dt.strftime('%H:%M'), rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

# Atur sumbu Y agar hanya menampilkan angka bulat
y_min = int(future_df['Predicted Index'].min())
y_max = int(future_df['Predicted Index'].max())
plt.yticks(np.arange(y_min, y_max + 1, step=1))
plt.legend()
plt.show()
