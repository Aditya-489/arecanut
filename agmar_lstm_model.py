import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib # Import joblib

# --- 1. Load and Prepare Data ---
df = pd.read_csv("arecanut.csv")
df['Price Date'] = pd.to_datetime(df['Price Date'])
df.sort_values('Price Date', inplace=True)
price_data = df[['Modal Price (Rs./Quintal)']].values

# --- 2. Scale the Data ---
# We scale the data first to a range of 0-1, which is optimal for LSTMs
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# --- 3. Create Training Sequences ---
# We will use the past 30 days (SEQUENCE_LENGTH) to predict the next day
SEQUENCE_LENGTH = 30
X, y = [], []

for i in range(len(scaled_data) - SEQUENCE_LENGTH):
    X.append(scaled_data[i:(i + SEQUENCE_LENGTH), 0])
    y.append(scaled_data[i + SEQUENCE_LENGTH, 0])

X, y = np.array(X), np.array(y)

# Reshape X to be [samples, timesteps, features] which is required for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# --- 4. Build and Train the LSTM Model ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
print("Training model...")
model.fit(X, y, epochs=25, batch_size=32)


# --- 5. Save the Model AND the Scaler ---
model.save('model.h5')
joblib.dump(scaler, 'scaler.gz') # Save the scaler for the app to use

print("\nSuccessfully trained and saved 'model.h5' and 'scaler.gz'")