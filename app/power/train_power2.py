import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.regularizers import l2
import joblib
import optuna
from optuna.integration import TFKerasPruningCallback
import gc
from tensorflow.keras.callbacks import ModelCheckpoint
from itertools import combinations
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = "data/activity_data_raw.csv"
PLANNED_WORKOUTS_PATH = "data/planned_workouts.csv"
TARGET_FEATURES = ["power_5s_avg"]
FEATURES = ["grade", "ascent_meters", "distance_meters", "atl_start", "ctl_start", "watt_kg", "temperature", "distance_diff", "grade_5s_avg", "grade_diff", "grade_30s_avg"]
ALL_FEATURES = TARGET_FEATURES + FEATURES
MODEL_PATH = "models/LSTM_power.keras"
METRICS_PATH = "data/model_metrics_power.csv"
SCALER_PATH = "models/scaler_power.pkl"

augmentation_methods = ['noise', 'scaling', 'shifting', 'time_warping']
all_augmentation_combinations = [tuple(comb) for comb in sum([list(combinations(augmentation_methods, i)) for i in range(1, len(augmentation_methods) + 1)], [])]

def load_activity_data(csv_path):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_data(df, drop_activities_with_workout=False):
    logger.info("Starting data preprocessing")
    logger.info(f"Initial shape: {df.shape}")
    
    df = df.sort_values(by=["activity_id", "timestamp"])
    df["altitude_diff"] = df["altitude"].diff()
    df["distance_diff"] = df["distance"].diff()
    df["ascent_meters"] = df["altitude_diff"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["grade_diff"] = df["grade"].diff().fillna(0)
    df["seconds_from_start"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df.rename(columns={"distance": "distance_meters"}, inplace=True)
    
    logger.info("Interpolating data...")
    df = df.groupby('activity_id').apply(lambda x: x.infer_objects().interpolate(method='linear')).reset_index(drop=True)
    logger.info("Filling NaN values...")
    df = df.ffill().bfill()

    df["power_5s_avg"] = df["power"].rolling(window=5, min_periods=1).mean()
    df["grade_5s_avg"] = df["grade"].rolling(window=5, min_periods=1).mean()
    df["grade_30s_avg"] = df["grade"].rolling(window=30, min_periods=1).mean()

    logger.info(f"Shape after preprocessing: {df.shape}")
    
    if df.isnull().sum().sum() > 0:
        logger.warning(f"There are still NaN values in the dataset: {df.isnull().sum()}")
    
    return df


def augment_data(seq, aug_methods):
    method = np.random.choice(aug_methods)
    if method == 'noise':
        noise = np.random.normal(0, 0.01, seq.shape)
        return seq + noise
    elif method == 'scaling':
        factor = np.random.uniform(0.9, 1.1)
        return seq * factor
    elif method == 'shifting':
        shift = np.random.randint(-3, 3)
        return np.roll(seq, shift, axis=0)
    elif method == 'time_warping':
        time_steps = np.arange(seq.shape[0])
        warp_steps = time_steps + np.random.normal(0, 1, time_steps.shape)
        warp_steps = np.clip(warp_steps, 0, len(time_steps) - 1)
        warped_seq = np.array([np.interp(time_steps, warp_steps, seq[:, i]) for i in range(seq.shape[1])]).T
        return warped_seq
    return seq

def create_sequences(data, sequence_length, target_columns, augmentation=None):
    num_records = len(data)
    num_features = len(FEATURES)
    num_targets = len(target_columns)

    sequences = np.zeros((num_records - sequence_length, sequence_length, num_features), dtype=np.float32)
    targets = np.zeros((num_records - sequence_length, num_targets), dtype=np.float32)

    feature_data = data[FEATURES].values
    target_data = data[target_columns].values

    for i in range(num_records - sequence_length):
        seq = feature_data[i:i + sequence_length]
        if augmentation:
            seq = augment_data(seq, augmentation)
        sequences[i] = seq
        targets[i] = target_data[i + sequence_length]

    return sequences, targets

def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

def build_model(
        learning_rate=0.001,
        cnn_filters=(),
        dense_units=0,
        dense_layers=0,
        lstm_units=0,
        lstm_layers=0,
        sequence_length=60,
        dropout_rate_dense=0, 
        dropout_rate_lstm=0,
        dropout_rate_cnn=0,
        add_batch_norm=False,
        l2_reg=0.01,
):
    model = Sequential()

    # CNN LAYERS
    if cnn_filters:
        for i, num_filters in enumerate(cnn_filters):
            if i == 0:
                model.add(Conv1D(filters=num_filters, kernel_size=3, activation='relu', input_shape=(sequence_length, len(FEATURES))))
            else:
                model.add(Conv1D(filters=num_filters, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_rate_cnn))
        if add_batch_norm:
            model.add(BatchNormalization())

    # LSTM LAYERS
    for i in range(lstm_layers):
        model.add(LSTM(lstm_units, return_sequences=(i < lstm_layers - 1), kernel_regularizer=l2(l2_reg)))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_lstm))

    # DENSE LAYERS
    for i in range(dense_layers):
        model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_dense))

    model.add(Dense(len(TARGET_FEATURES)))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    
    logger.info("LSTM model built and compiled")
    return model

def objective(trial, train_df, val_df, test_df):
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    sequence_length = trial.suggest_int('sequence_length', 10, 130)
    lstm_units = trial.suggest_int('lstm_units', 32, 256)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 10)
    dense_units = trial.suggest_int('dense_units', 32, 256)
    dense_layers = trial.suggest_int('dense_layers', 1, 10)
    dropout_rate_lstm = trial.suggest_float('dropout_rate_lstm', 0.0, 0.5)
    dropout_rate_dense = trial.suggest_float('dropout_rate_dense', 0.0, 0.5)
    dropout_rate_cnn = trial.suggest_float('dropout_rate_cnn', 0.0, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 0.0, 0.1)
    cnn_filters = trial.suggest_categorical('cnn_filters', [(), (32, 64), (64, 128), (128, 64), (64, 32)])
    augmentation_methods = trial.suggest_categorical('augmentation_methods', all_augmentation_combinations)
    scaler_type = trial.suggest_categorical('scaler_type', ('MinMaxScaler', 'StandardScaler'))

    model = build_model(
        learning_rate=learning_rate,
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        dense_units=dense_units,
        dense_layers=dense_layers,
        dropout_rate_lstm=dropout_rate_lstm,
        dropout_rate_dense=dropout_rate_dense,
        dropout_rate_cnn=dropout_rate_cnn,
        l2_reg=l2_reg,
        sequence_length=sequence_length
    )

    batch_size = 1024
    checkpoint_callback = ModelCheckpoint(filepath=f'checkpoint_trial_{trial.number}.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
    
    train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler = scale_data(train_df, val_df, test_df, scaler_type)

    train_sequences, train_targets_seq = create_sequences(train_scaled, sequence_length, TARGET_FEATURES, augmentation_methods)
    val_sequences, val_targets_seq = create_sequences(val_scaled, sequence_length, TARGET_FEATURES)

    history = model.fit(train_sequences, train_targets_seq, batch_size=batch_size, epochs=50, validation_data=(val_sequences, val_targets_seq), callbacks=[
        TFKerasPruningCallback(trial, 'val_loss'),
        LearningRateScheduler(scheduler),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        checkpoint_callback
    ], verbose=0)

    val_loss = min(history.history['val_loss'])

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return val_loss

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    mae = np.mean(np.abs(predictions - y))
    mape = np.mean(np.abs((y - predictions) / y)) * 100
    return {'MSE': mse, 'MAE': mae, 'MAPE': mape}

def split_by_activities(df):
    if df is None or df.empty:
        logger.error("DataFrame is None or empty in split_by_activities")
        return None, None, None
    
    activity_ids = df['activity_id'].unique()
    train_ids, test_ids = train_test_split(activity_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1 / 0.8, random_state=42)
    
    train_df = df[df['activity_id'].isin(train_ids)]
    val_df = df[df['activity_id'].isin(val_ids)]
    test_df = df[df['activity_id'].isin(test_ids)]
    
    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Validation set shape: {val_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")
    
    return train_df, val_df, test_df

def scale_data(train, val, test, scaler_type="StandardScaler"):
    if scaler_type == "StandardScaler":
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

    feature_scaler.fit(train[FEATURES])  
    target_scaler.fit(train[TARGET_FEATURES])  

    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    train_scaled[FEATURES] = feature_scaler.transform(train[FEATURES])
    val_scaled[FEATURES] = feature_scaler.transform(val[FEATURES])
    test_scaled[FEATURES] = feature_scaler.transform(test[FEATURES])

    train_scaled[TARGET_FEATURES] = target_scaler.transform(train[TARGET_FEATURES])
    val_scaled[TARGET_FEATURES] = target_scaler.transform(val[TARGET_FEATURES])
    test_scaled[TARGET_FEATURES] = target_scaler.transform(test[TARGET_FEATURES])
    
    return train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler

def main():
    df = load_activity_data(RAW_DATA_PATH)
    if df is None or df.empty:
        logger.error("Failed to load data")
        return

    df = preprocess_data(df, drop_activities_with_workout=True)
    if df is None or df.empty:
        logger.error("Failed to preprocess data")
        return

    train_df, val_df, test_df = split_by_activities(df)
    if train_df is None or val_df is None or test_df is None:
        logger.error("Failed to split data")
        return

    study_name = "power_study"  
    storage_name = "sqlite:///power_study.db"  

    study = optuna.create_study(study_name=study_name, direction='minimize', storage=storage_name, load_if_exists=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    try:
        study.optimize(lambda trial: objective(trial, train_df, val_df, test_df), n_trials=50)  
    except Exception as e:
        print(f"Optimization failed with exception: {e}")

    if len(study.trials) > 0 and study.best_trial is not None:
        best_trial = study.best_trial
        results = {
            'learning_rate': best_trial.params['learning_rate'],
            'sequence_length': best_trial.params['sequence_length'],
            'lstm_units': best_trial.params['lstm_units'],
            'lstm_layers': best_trial.params['lstm_layers'],
            'dense_units': best_trial.params['dense_units'],
            'dense_layers': best_trial.params['dense_layers'],
            'dropout_rate_lstm': best_trial.params['dropout_rate_lstm'],
            'dropout_rate_dense': best_trial.params['dropout_rate_dense'],
            'dropout_rate_cnn': best_trial.params['dropout_rate_cnn'],
            'l2_reg': best_trial.params['l2_reg'],
            'cnn_filters': best_trial.params['cnn_filters'],
            'augmentation_methods': best_trial.params['augmentation_methods'],
            'scaler_type': best_trial.params['scaler_type'],
            'val_loss': best_trial.value
        }

        results_df = pd.DataFrame([results])
        results_df.to_csv(METRICS_PATH, index=False)

        train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler = scale_data(train_df, val_df, test_df, results['scaler_type'])

        train_sequences, train_targets = create_sequences(train_scaled, results['sequence_length'], TARGET_FEATURES, results['augmentation_methods'])
        val_sequences, val_targets = create_sequences(val_scaled, results['sequence_length'], TARGET_FEATURES)

        model = build_model(
            learning_rate=results['learning_rate'],
            cnn_filters=results['cnn_filters'],
            lstm_units=results['lstm_units'],
            lstm_layers=results['lstm_layers'],
            dense_units=results['dense_units'],
            dense_layers=results['dense_layers'],
            dropout_rate_lstm=results['dropout_rate_lstm'],
            dropout_rate_dense=results['dropout_rate_dense'],
            dropout_rate_cnn=results['dropout_rate_cnn'],
            l2_reg=results['l2_reg'],
            sequence_length=results['sequence_length']
        )

        model.summary()

        BATCH_SIZE = 1024
        EPOCHS = 50
        PATIENCE = 5

        history = model.fit(train_sequences, train_targets, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_sequences, val_targets), callbacks=[
            LearningRateScheduler(scheduler),
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        ], verbose=2)

        test_sequences, test_targets = create_sequences(test_scaled, results['sequence_length'], TARGET_FEATURES)
        test_metrics = evaluate_model(model, test_sequences, test_targets)
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")

        model.save(MODEL_PATH)
        joblib.dump(feature_scaler, "models/feature_scaler.pkl")
        joblib.dump(target_scaler, "models/target_scaler.pkl")

        # SHAP values calculation
        explainer = shap.DeepExplainer(model, train_sequences[:100])
        shap_values = explainer.shap_values(test_sequences[:100])
        
        shap.summary_plot(shap_values[0], test_sequences[:100], feature_names=FEATURES, show=False)
        plt.savefig('shap_summary_plot.png')
        plt.close()

if __name__ == "__main__":
    main()