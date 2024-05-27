import sqlite3
from io import StringIO
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import os

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = "data/activity_data_raw.csv"
SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 50
TARGET_FEATURES = ["power", "seconds_from_start"]
FEATURES = ["grade", "ascent_meters", "distance_meters", "atl_start", "ctl_start", "watt_kg", "temperature"]
ALL_FEATURES = TARGET_FEATURES + FEATURES
MODEL_PATH = "models/LSTM_power"
METRICS_PATH = "data/model_metrics_power.csv"

def load_activity_data(csv_path):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Data loaded successfully")

    return df

def preprocess_data(df):
    logger.info("Starting data preprocessing")
    df = df.sort_values(by=["activity_id", "timestamp"])
    df["altitude_diff"] = df["altitude"].diff().fillna(0)
    df["ascent_meters"] = df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["seconds_from_start"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df.rename(columns={"distance": "distance_meters"}, inplace=True)
    scaler = StandardScaler()
    df[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])
    logger.info("Data preprocessing completed")
    return df

def create_sequences(data, sequence_length, target_columns):
    logger.info("Creating sequences for the model")
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length][FEATURES].values.astype('float32')
        target = data.iloc[i + sequence_length][target_columns].values.astype('float32')
        sequences.append(seq)
        targets.append(target)
    logger.info("Sequences created successfully")
    return np.array(sequences), np.array(targets)

def build_model(input_shape, output_shape):
    logger.info("Building the LSTM model")
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(output_shape))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    logger.info("LSTM model built and compiled")
    return model

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_test[:, 0], label='True Power')
    plt.plot(y_pred[:, 0], label='Predicted Power')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(y_test[:, 1], label='True Seconds from Start')
    plt.plot(y_pred[:, 1], label='Predicted Seconds from Start')
    plt.legend()
    plt.show()

def save_model(model, path):
    logger.info(f"Saving model to {path}")
    model.save(path)
    logger.info("Model saved successfully")

def save_metrics(metrics, path):
    logger.info(f"Saving metrics to {path}")
    df_metrics = pd.DataFrame([metrics])
    if os.path.exists(path):
        df_metrics.to_csv(path, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(path, index=False)
    logger.info("Metrics saved successfully")

def main():

    df = load_activity_data(RAW_DATA_PATH)
    df = preprocess_data(df)

    all_sequences, all_targets = [], []
    for activity_id in df["activity_id"].unique():
        activity_data = df[df["activity_id"] == activity_id]
        sequences, targets = create_sequences(activity_data, SEQ_LENGTH, TARGET_FEATURES)
        all_sequences.append(sequences)
        all_targets.append(targets)

    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_targets, axis=0)

    logger.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((SEQ_LENGTH, len(FEATURES)), len(TARGET_FEATURES))
    logger.info(model.summary())

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, valactivity_idation_data=(X_test, y_test))

    val_loss = model.evaluate(X_test, y_test)
    logger.info(f'Valactivity_idation Loss: {val_loss}')

    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics = {
        'model': 'LSTM_power',
        'r2_score': r2,
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'timestamp': timestamp
    }

    logger.info(f"Metrics: R2 Score: {r2}, MAE: {mae}, MSE: {mse}")

    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)

    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
