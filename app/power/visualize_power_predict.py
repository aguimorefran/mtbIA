import os
import logging
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import gpxpy
import haversine as hs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "models/LSTM_power.keras"
FEATURE_SCALER_PATH = "models/feature_scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"
BEST_PARAMS_PATH = "models/best_params.pkl"
FEATURES = ["grade", "ascent_meters", "distance_meters", "atl_start", "ctl_start", "watt_kg", "temperature", "distance_diff", "grade_5s_avg", "grade_diff", "grade_30s_avg"]
TARGET_FEATURES = ["power_5s_avg"]

def load_best_model_and_params():
    """Load the best model, scalers, and parameters."""
    try:
        model = load_model(MODEL_PATH)
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        best_params = joblib.load(BEST_PARAMS_PATH)
        return model, feature_scaler, target_scaler, best_params
    except Exception as e:
        logger.error(f"Error loading model and parameters: {e}")
        raise

def read_gpx(gpx_path):
    """Read GPX file and return DataFrame."""
    try:
        with open(gpx_path, "rb") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        data.append([point.latitude, point.longitude, point.elevation, point.time])
        df = pd.DataFrame(data, columns=["latitude", "longitude", "altitude", "time"])
        return df
    except Exception as e:
        logger.error(f"Failed to read GPX file: {e}")
        raise

def calculate_distance_slope(activity_df):
    """Calculate distance and slope for each point in the activity."""
    activity_df = activity_df.copy()
    activity_df["distance"] = 0.0
    for i in range(1, len(activity_df)):
        distance = hs.haversine(
            (activity_df.loc[i - 1, "latitude"], activity_df.loc[i - 1, "longitude"]),
            (activity_df.loc[i, "latitude"], activity_df.loc[i, "longitude"]),
            unit=hs.Unit.METERS,
        )
        activity_df.loc[i, "distance"] = distance
    activity_df["distance"] = activity_df["distance"].fillna(0)
    activity_df["altitude_diff"] = activity_df["altitude"].diff().fillna(0)
    diffs = activity_df["distance"].replace(0, np.nan)
    slopes = np.where(diffs != 0, activity_df["altitude_diff"] / diffs * 100, 0)
    max_slope = 100
    slopes = np.clip(slopes, -max_slope, max_slope)
    activity_df["slope"] = slopes
    activity_df["slope"] = activity_df["slope"].fillna(0)
    activity_df["distance_diff"] = activity_df["distance"].diff().fillna(0)
    activity_df["distance_accumulated"] = activity_df["distance"].cumsum()
    return activity_df

def preprocess_activity_data(activity_df, wellness_data):
    """Preprocess activity data for prediction."""
    activity_df = calculate_distance_slope(activity_df)
    activity_df["grade_5s_avg"] = activity_df["slope"].rolling(window=5, min_periods=1).mean()
    activity_df["grade_30s_avg"] = activity_df["slope"].rolling(window=30, min_periods=1).mean()
    activity_df["grade_diff"] = activity_df["slope"].diff().fillna(0)
    activity_df["ascent_meters"] = activity_df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    
    # Merge wellness data
    activity_df = pd.merge(activity_df, wellness_data, left_index=True, right_index=True, how='left')
    
    return activity_df[FEATURES]

def create_sequences(data, sequence_length):
    """Create sequences from the data."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

def predict_power(gpx_path, wellness_data):
    """Predict power for a given GPX file and wellness data."""
    model, feature_scaler, target_scaler, best_params = load_best_model_and_params()
    
    gpx_data = read_gpx(gpx_path)
    preprocessed_data = preprocess_activity_data(gpx_data, wellness_data)
    
    sequence_length = best_params['sequence_length']
    input_sequences = create_sequences(preprocessed_data, sequence_length)
    
    input_scaled = feature_scaler.transform(input_sequences.reshape(-1, input_sequences.shape[-1])).reshape(input_sequences.shape)
    
    predictions_scaled = model.predict(input_scaled)
    predictions = target_scaler.inverse_transform(predictions_scaled)
    
    return predictions

if __name__ == "__main__":
    # Example usage
    gpx_path = "path/to/your/activity.gpx"
    wellness_data = pd.DataFrame({
        'atl_start': [50],
        'ctl_start': [60],
        'watt_kg': [3.5],
        'temperature': [20]
    })
    
    try:
        power_predictions = predict_power(gpx_path, wellness_data)
        print("Power predictions:", power_predictions)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")