import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Constants
RAW_DATA_PATH = "data/activity_data_raw.csv"
MODEL_PATH = "models/LSTM_power.keras"
SCALER_PATH = "models/scaler_power.pkl"
TARGET_FEATURES = ["power_5s_avg"]
FEATURES = ["grade", "ascent_meters", "distance_meters", "atl_start", "ctl_start", "watt_kg", "temperature", "distance_diff", "grade_5s_avg", "grade_diff"]
ALL_FEATURES = TARGET_FEATURES + FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_activity_data(csv_path):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Data loaded successfully")
    return df

def preprocess_data(df):
    logger.info("Starting data preprocessing")
    df = df.sort_values(by=["activity_id", "timestamp"])
    df["altitude_diff"] = df["altitude"].diff()
    df["distance_diff"] = df["distance"].diff()
    df["ascent_meters"] = df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["grade_diff"] = df["grade"].diff().fillna(0)
    df["seconds_from_start"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df.rename(columns={"distance": "distance_meters"}, inplace=True)
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df["power_5s_avg"] = df["power"].rolling(window=5, min_periods=1).mean()
    df["grade_5s_avg"] = df["grade"].rolling(window=5, min_periods=1).mean()
    
    return df

def create_sequences(data, sequence_length, target_columns):
    num_records = len(data)
    num_features = len(FEATURES)
    num_targets = len(target_columns)

    sequences = np.zeros((num_records - sequence_length, sequence_length, num_features), dtype=np.float32)
    targets = np.zeros((num_records - sequence_length, num_targets), dtype=np.float32)

    feature_data = data[FEATURES].values
    target_data = data[target_columns].values

    for i in range(num_records - sequence_length):
        sequences[i] = feature_data[i:i + sequence_length]
        targets[i] = target_data[i + sequence_length]

    return sequences, targets

def predict_activity(activity_id, model, scaler, sequence_length):
    df = load_activity_data(RAW_DATA_PATH)
    df = preprocess_data(df)
    activity_df = df[df['activity_id'] == activity_id]
    
    if activity_df.empty:
        logger.error(f"Activity ID {activity_id} not found in the data.")
        return

    activity_df_scaled = activity_df.copy()
    activity_df_scaled[FEATURES] = scaler.transform(activity_df[FEATURES])

    sequences, targets = create_sequences(activity_df_scaled, sequence_length, TARGET_FEATURES)

    predictions = model.predict(sequences)
    return predictions, targets

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Reemplaza con un activity_id usado en el entrenamiento
    activity_id = 10896724321  
    sequence_length = 60  # Reemplaza con el valor correcto

    predictions, targets = predict_activity(activity_id, model, scaler, sequence_length)

    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i]}, Actual: {targets[i]}")
