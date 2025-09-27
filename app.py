from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import joblib # Import joblib

# --- Step 1: Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Step 2: Load Model and Scaler ---
MODEL_PATH = 'model.h5'
SCALER_PATH = 'scaler.gz'
model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model or scaler. Error: {e}")
else:
    print(f"FATAL ERROR: Make sure 'model.h5' and 'scaler.gz' are present.")

# --- Step 3: Define Constants ---
SEQUENCE_LENGTH = 30
CSV_PATH = 'Agmar.csv'

# --- Step 4: Function to Get Historical Data ---
def get_last_30_days_data():
    """Fetches the last 30 days of price data from the CSV."""
    if not os.path.exists(CSV_PATH):
        return None, f"'{CSV_PATH}' not found."

    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['Price Date'])
        df.rename(columns={'Price Date': 'date', 'Modal Price (Rs./Quintal)': 'price'}, inplace=True)
        
        if len(df) < SEQUENCE_LENGTH:
            return None, f"Not enough data. Need at least {SEQUENCE_LENGTH} rows."
            
        price_data = df.sort_values('date', ascending=True).tail(SEQUENCE_LENGTH)
        return price_data['price'].values, None
        
    except Exception as e:
        return None, f"Error reading data: {e}"

# --- Step 5: Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    past_prices, error_message = get_last_30_days_data()
    if error_message:
        return jsonify({'error': error_message}), 400

    try:
        # 1. Reshape and scale the input data
        past_prices_reshaped = past_prices.reshape(-1, 1)
        scaled_prices = scaler.transform(past_prices_reshaped)

        # 2. Reshape for the LSTM model [samples, timesteps, features]
        input_data = scaled_prices.reshape(1, SEQUENCE_LENGTH, 1)

        # 3. Make a prediction
        predicted_scaled_price = model.predict(input_data)

        # 4. Inverse transform the prediction to get the actual price
        predicted_price = scaler.inverse_transform(predicted_scaled_price)

        return jsonify({'predicted_price': round(float(predicted_price[0][0]), 2)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Step 6: Run the App ---
if __name__ == '__main__':
    if model is not None and scaler is not None:
        print("Starting Flask server...")
        app.run(debug=False, port=5001) # Set debug=False for cleaner output
    else:
        print("Server did not start because model/scaler could not be loaded.")