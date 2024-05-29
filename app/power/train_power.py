import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import os

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = "data/activity_data_raw.csv"
PLANNED_WORKOUTS_PATH = "data/planned_workouts.csv"
SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 50
TARGET_FEATURES = ["power_3s_avg", "seconds_from_start"]
FEATURES = ["grade", "ascent_meters", "distance_meters", "atl_start", "ctl_start", "watt_kg", "temperature"]
ALL_FEATURES = TARGET_FEATURES + FEATURES
MODEL_PATH = "models/LSTM_power.keras"
METRICS_PATH = "data/model_metrics_power.csv"

def load_activity_data(csv_path):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Data loaded successfully")
    return df

def preprocess_data(df, drop_activities_with_workout=False):
    logger.info("Starting data preprocessing")
    df = df.sort_values(by=["activity_id", "timestamp"])
    df["altitude_diff"] = df["altitude"].diff().fillna(0)
    df["ascent_meters"] = df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["seconds_from_start"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df.rename(columns={"distance": "distance_meters"}, inplace=True)
    scaler = StandardScaler()

    # Fill NaN values with a middle value between the previous and next values
    df.fillna(method='ffill', inplace=True)

    # Calculate 3-second moving average for power
    df["power_3s_avg"] = df["power"].rolling(window=3, min_periods=1).mean()

    # Load planned workouts
    planned_workouts = pd.read_csv(PLANNED_WORKOUTS_PATH)

    # If drop_activities_with_workout is True, drop activities whose id appear in planned_workouts["paired_activity_id"]
    if drop_activities_with_workout:
        intersecting_activities = planned_workouts["paired_activity_id"].unique()
        logger.info(f"Dropping {len(intersecting_activities)} activities with planned workouts")
        df = df[~df["activity_id"].isin(intersecting_activities)]
        logger.info(f"Remaining activities: {df['activity_id'].nunique()}")

    df[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])

    rows_na = df.isna().sum().sum()
    df.dropna(inplace=True)
    logger.info(f"Dropped {rows_na} rows with NaN values")
    if df[ALL_FEATURES].isnull().values.any() or np.isinf(df[ALL_FEATURES]).values.any():
        logger.error("NaN or infinite values found in the data")
        raise ValueError("NaN or infinite values found in the data")

    logger.info("Data preprocessing completed")
    return df

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

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


def build_model(units=100, dropout_rate=0.2, learning_rate=0.001):
    logger.info(
        f"Building the LSTM model with units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
    model = Sequential()
    model.add(LSTM(units, activation="relu", input_shape=(SEQ_LENGTH, len(FEATURES)), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, activation="relu", return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(TARGET_FEATURES)))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
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
    ensure_dir(path)
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
    df = preprocess_data(df, drop_activities_with_workout=True)

    logger.info(f"Creating sequences with length {SEQ_LENGTH}")
    all_sequences, all_targets = [], []
    for id in df["activity_id"].unique():
        activity_data = df[df["activity_id"] == id]
        sequences, targets = create_sequences(activity_data, SEQ_LENGTH, TARGET_FEATURES)
        all_sequences.append(sequences)
        all_targets.append(targets)

    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_targets, axis=0)

    logger.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the best hyperparameters found
    best_hyperparams = {'units': 50, 'dropout_rate': 0.2, 'learning_rate': 0.001}
    logger.info(f"Training with best hyperparameters: {best_hyperparams}")

    model = build_model(**best_hyperparams)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    val_loss = model.evaluate(X_test, y_test)
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

