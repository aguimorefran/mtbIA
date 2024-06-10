import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import RNN, LSTMCell
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping


# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = "data/activity_data_raw.csv"
PLANNED_WORKOUTS_PATH = "data/planned_workouts.csv"
TARGET_FEATURES = ["power_3s_avg"]
FEATURES = ["grade", "ascent_meters", "distance_meters", "atl_start", "ctl_start", "watt_kg", "temperature",
            "distance_diff"]
ALL_FEATURES = TARGET_FEATURES + FEATURES
MODEL_PATH = "models/LSTM_power.keras"
METRICS_PATH = "data/model_metrics_power.csv"


def load_activity_data(csv_path):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Data loaded successfully")
    return df


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_data(df, drop_activities_with_workout=False):
    logger.info("Starting data preprocessing")
    df = df.sort_values(by=["activity_id", "timestamp"])
    df["altitude_diff"] = df["altitude"].diff()
    df["distance_diff"] = df["distance"].diff()
    df["ascent_meters"] = df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["seconds_from_start"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df.rename(columns={"distance": "distance_meters"}, inplace=True)

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df["power_3s_avg"] = df["power"].rolling(window=3, min_periods=1).mean()

    planned_workouts = pd.read_csv(PLANNED_WORKOUTS_PATH)

    if drop_activities_with_workout:
        intersecting_activities = planned_workouts["paired_activity_id"].unique()
        logger.info(f"Dropping {len(intersecting_activities)} activities with planned workouts")
        df = df[~df["activity_id"].isin(intersecting_activities)]
        logger.info(f"Remaining activities: {df['activity_id'].nunique()}")

    rows_na = df.isna().sum().sum()
    df.dropna(inplace=True)
    logger.info(f"Dropped {rows_na} rows with NaN values")
    if df[ALL_FEATURES].isnull().values.any() or np.isinf(df[ALL_FEATURES]).values.any():
        logger.error("NaN or infinite values found in the data")
        raise ValueError("NaN or infinite values found in the data")

    logger.info("Data preprocessing completed")
    return df


def create_sequences(data, sequence_length, target_columns, n_activities=None):
    if n_activities is not None:
        unique_activities = data['activity_id'].unique()
        if n_activities > len(unique_activities):
            raise ValueError("n_activities cannot be greater than the number of unique activities in the data")
        selected_activities = np.random.choice(unique_activities, size=n_activities, replace=False)
        data = data[data['activity_id'].isin(selected_activities)]

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


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
def build_model(units=50, dropout_rate=0.2, learning_rate=0.001, dense_neurons=8, dense_layers=4, lstm_layers=3, sequence_length=300):
    logger.info(f"Building the LSTM model with units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
    model = Sequential()

    # Add dense layers
    for _ in range(dense_layers):
        model.add(Dense(dense_neurons, activation='relu', input_shape=(sequence_length, len(FEATURES))))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Add LSTM layers
    for i in range(lstm_layers):
        return_seq = True if i < lstm_layers - 1 else False
        model.add(RNN(
            LSTMCell(units, activation="tanh", use_bias=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform'),
            input_shape=(sequence_length, len(FEATURES)),
            return_sequences=return_seq
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(len(TARGET_FEATURES)))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    logger.info("LSTM model built and compiled")
    return model

def get_model_metrics(model, sequences, targets):
    logger.info("Getting model metrics")
    predictions = model.predict(sequences)
    metrics = {}
    for i, target in enumerate(TARGET_FEATURES):
        metric = tf.keras.metrics.mean_squared_error(targets[:, i], predictions[:, i]).numpy()
        metrics[target] = metric
    logger.info(f"Model metrics: {metrics}")
    return metrics

def plot_history(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

def main():
    LR_INIT = 0.001
    SEQ_LENGTH = 300
    BATCH_SIZE = 5120
    EPOCHS = 50
    ACTIVITIES_TO_USE = 5
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(e)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    df = load_activity_data(RAW_DATA_PATH)
    df = preprocess_data(df, drop_activities_with_workout=True)

    # Get the sequences of 1 activity
    sequences, targets = create_sequences(df, SEQ_LENGTH, TARGET_FEATURES, n_activities=ACTIVITIES_TO_USE)

    scaler = StandardScaler()
    for i in range(sequences.shape[0]): # Scale each sequence
        sequences[i, :, :] = scaler.fit_transform(sequences[i, :, :])


    targets = scaler.fit_transform(targets)


    # Train a model with the sequences
    lr_scheduler = LearningRateScheduler(scheduler)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    model = build_model(dense_layers=0, dense_neurons=0, learning_rate=LR_INIT, sequence_length=SEQ_LENGTH)
    history = model.fit(sequences, targets, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])

    plot_history(history)



if __name__ == "__main__":
    main()
