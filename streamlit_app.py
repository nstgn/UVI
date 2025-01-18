# 1. Import Library
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 2. Konfigurasi Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    r"E:\A) SEMESTER 9\Code TA\.streamlit\uv-gsheets-439900-f7eafe9b374b.json", scope
)
client = gspread.authorize(creds)
sheet_url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit"
worksheet = client.open_by_url(sheet_url).sheet1

# 3. Baca Data dari Google Sheets
data = pd.DataFrame(worksheet.get_all_records())
print("Data awal:\n", data.head())  # Validasi struktur data

# 4. Data Preprocessing
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('Datetime', inplace=True)
data = data[['Index']].copy()

# Resampling data setiap 2 menit dan interpolasi
data = data.resample('2T').mean()
data['Index'].interpolate(method='linear', inplace=True)

# 5. Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Index_scaled'] = scaler.fit_transform(data[['Index']])

# 6. Persiapan Data untuk LSTM
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 7
X, y = prepare_data(data['Index_scaled'].values, n_steps)

# Split data menjadi train dan test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Visualisasi Data Train dan Test
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), y_train, label='Data Train', color='purple')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Data Test', color='green')
plt.title("Data Train dan Data Test")
plt.xlabel("Index")
plt.ylabel("Scaled Intensity")
plt.legend()
plt.show()

# 7. Bangun Model LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])

# Kompilasi Model
model.compile(optimizer='adam', loss='mean_squared_error')

# 8. Pelatihan Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 9. Prediksi dan Evaluasi
train_predicted = model.predict(X_train)
test_predicted = model.predict(X_test)

# Evaluasi untuk data training
mse_train = mean_squared_error(y_train, train_predicted)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_predicted)
r2_train = r2_score(y_train, train_predicted) * 100

# Evaluasi untuk data testing
mse_test = mean_squared_error(y_test, test_predicted)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_predicted)
r2_test = r2_score(y_test, test_predicted) * 100

# Menampilkan hasil evaluasi
print("Training Metrics:")
print(f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.2f}%")
print("\nTesting Metrics:")
print(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.2f}%")

# 10. Visualisasi Prediksi vs Aktual
sns.set_style("darkgrid")
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
train_predicted = scaler.inverse_transform(train_predicted)
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
test_predicted = scaler.inverse_transform(test_predicted)

plt.figure(figsize=(10, 6))
plt.plot(range(len(train_actual)), train_actual, label='Actual Train', color='blue')
plt.plot(range(len(train_actual), len(train_actual) + len(test_actual)), test_actual, label='Actual Test', color='green')
plt.plot(range(len(train_predicted)), train_predicted, label='Predicted Train', color='red')
plt.plot(range(len(train_predicted), len(train_predicted) + len(test_predicted)), test_predicted, label='Predicted Test', color='orange')
plt.title('Prediksi vs Aktual (Latih dan Uji)')
plt.xlabel('Time')
plt.ylabel('UV Index')
plt.legend()
plt.show()
