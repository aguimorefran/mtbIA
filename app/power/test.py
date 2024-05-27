import sqlite3
from io import StringIO
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "../data/activities_blob.db"
SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 50
TARGET_FEATURES = ["power", "seconds_from_start"]
FEATURES = ["grade", "ascent_meters", "distance_meters"]
ALL_FEATURES = TARGET_FEATURES + FEATURES

def load_activity_data(db_path):
    logger.info(f"Cargando datos desde {db_path}")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM activities")
        data = cursor.fetchone()
        df = pd.read_json(StringIO(data[1]))
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["id"] = data[0]
    logger.info("Datos cargados correctamente")
    return df

def preprocess_data(df):
    logger.info("Iniciando el preprocesamiento de los datos")
    df = df.sort_values(by=["id", "timestamp"])
    df["altitude_diff"] = df["altitude"].diff().fillna(0)
    df["ascent_meters"] = df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["seconds_from_start"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df.rename(columns={"distance": "distance_meters"}, inplace=True)
    scaler = StandardScaler()
    df[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])
    logger.info("Preprocesamiento completado")
    return df

def create_sequences(data, sequence_length, target_columns):
    """Crea secuencias para el entrenamiento del modelo."""
    logger.info("Creando secuencias para el modelo")
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length][FEATURES].values.astype('float32')
        target = data.iloc[i + sequence_length][target_columns].values.astype('float32')
        sequences.append(seq)
        targets.append(target)
    logger.info("Secuencias creadas correctamente")
    return np.array(sequences), np.array(targets)

def build_model(input_shape, output_shape):
    """Construye y compila el modelo LSTM."""
    logger.info("Construyendo el modelo LSTM")
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(output_shape))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    logger.info("Modelo LSTM construido y compilado")
    return model

def plot_predictions(y_test, y_pred):
    """Grafica las predicciones frente a los valores reales."""
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

def main():
    df = load_activity_data(DB_PATH)
    df = preprocess_data(df)

    all_sequences, all_targets = [], []
    for id in df["id"].unique():
        activity_data = df[df["id"] == id]
        sequences, targets = create_sequences(activity_data, SEQ_LENGTH, TARGET_FEATURES)
        all_sequences.append(sequences)
        all_targets.append(targets)

    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_targets, axis=0)

    logger.info(f"Shape de X: {X.shape}, Shape de y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((SEQ_LENGTH, len(FEATURES)), len(TARGET_FEATURES))
    logger.info(model.summary())

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

    val_loss = model.evaluate(X_test, y_test)
    logger.info(f'Validation Loss: {val_loss}')

    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
