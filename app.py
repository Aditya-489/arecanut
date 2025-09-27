
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
# import pickle # No longer needed
import numpy as np
import pandas as pd
from flask_cors import CORS

# --- Step 1: Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Allows the frontend to communicate with this backend

# --- Step 2: Load Your Model ONLY ---
# The scaler is no longer loaded.
try:
    # MODIFICATION: Added compile=False to handle version mismatch errors.
    model = load_model('model.h5', compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    
# LSTM model requires a sequence of past data, e.g., 30 days.
SEQUENCE_LENGTH = 30

# --- Step 3: Create a Function to Get Historical Data ---
def get_last_30_days_data():
    """
    Fetches the last 30 days of price data from the CSV.
    """
    print("Fetching historical data...")
    try:
        # Reading from 'Agmar.csv'
        df = pd.read_csv('Agmar.csv', parse_dates=['date'])
        
        # Get the last 30 entries sorted by date
        price_data = df.sort_values('date', ascending=False).head(SEQUENCE_LENGTH)
        
        if len(price_data) < SEQUENCE_LENGTH:
            return None
            
        # Return the prices as a list (oldest to newest for the model)
        return price_data['price'].to_list()[::-1]
        
    except FileNotFoundError:
        print("ERROR: 'Agmar.csv' not found. Please make sure the file is in the same folder as app.py.")
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# --- Step 4: Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Check backend logs.'}), 500

    try:
        # 1. Get the historical data
        past_prices = get_last_30_days_data()
        
        if past_prices is None:
             return jsonify({'error': 'Not enough historical data to make a prediction. Check Agmar.csv.'}), 400

        # 2. Preprocess the data for the model (WITHOUT SCALING)
        input_data = np.array(past_prices).reshape(-1, 1)
        # The 'scaler.transform' step is removed.
        final_input = input_data.reshape(1, SEQUENCE_LENGTH, 1) 

        # 3. Make a prediction
        prediction = model.predict(final_input)

        # 4. Get the result (WITHOUT INVERSE TRANSFORM)
        # The 'scaler.inverse_transform' step is removed.
        predicted_price = prediction

        # Return the prediction as JSON
        return jsonify({'predicted_price': round(float(predicted_price[0][0]), 2)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Step 5: Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

