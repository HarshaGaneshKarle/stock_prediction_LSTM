import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import datetime
from tensorflow.keras.callbacks import EarlyStopping

# 1. Download MSFT stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Define date range
start_date = '2015-01-01'
end_date = '2025-07-12'
ticker = 'MSFT'

# Fetch data
df = fetch_stock_data(ticker, start_date, end_date)
print("Data fetched successfully:")
print(df.head())

# 2. Prepare data (use 'Close' price for prediction)
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM (e.g., use past 60 days to predict next day)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Split into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 3. Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, 
                    verbose=1, callbacks=[early_stopping])


# Summary of the model
model.summary()

# 4. Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# 5. Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# 6. Plot results
plt.figure(figsize=(14, 7))
plt.plot(df.index[sequence_length:train_size + sequence_length], train_predict, label='Train Predictions')
plt.plot(df.index[train_size + sequence_length:], test_predict, label='Test Predictions')
plt.plot(df.index[sequence_length:], scaler.inverse_transform(scaled_data[sequence_length:]), label='Actual Prices')
plt.title('MSFT Stock Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 7. Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8. Predict future trends (e.g., next 30 days)
last_sequence = scaled_data[-sequence_length:]
future_predictions = []
future_days = 30

for _ in range(future_days):
    last_sequence_reshaped = last_sequence.reshape((1, sequence_length, 1))
    next_pred = model.predict(last_sequence_reshaped)
    future_predictions.append(next_pred[0, 0])
    last_sequence = np.append(last_sequence[1:], next_pred, axis=0)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates
last_date = df.index[-1]
future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, future_days + 1)]

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(df.index[-100:], scaler.inverse_transform(scaled_data[-100:]), label='Historical Prices')
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
plt.title('MSFT Stock Price Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()