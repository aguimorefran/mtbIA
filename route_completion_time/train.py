import pandas as pd
import os
import pickle
import logging
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/activity_data.csv"
MODEL_METRICS_SAVE_PATH = "data/model_metrics.csv"
MODELS_SAVE_PATH = "models/"
SCALER_SAVE_PATH = "models/scaler.pkl"

PREDICT_FEATURE = "duration_seconds"
IGNORE_COLUMNS = ["activity_id", "date"]


def ensure_directories():
    os.makedirs(os.path.dirname(MODEL_METRICS_SAVE_PATH), exist_ok=True)
    os.makedirs(MODELS_SAVE_PATH, exist_ok=True)


def save_metrics(path, metrics):
    df = pd.DataFrame(metrics)
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    logging.info("Metrics saved to %s", path)


def save_model(path, model):
    with open(path, "wb") as model_file:
        pickle.dump(model, model_file)
    logging.info("Model saved to %s", path)


def save_scaler(path, scaler):
    with open(path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    logging.info("Scaler saved to %s", path)


def process_data(data_path, predict_feature, ignore_columns):
    df = pd.read_csv(data_path).dropna()
    total_rows = df.shape[0]
    X = df.drop(columns=[predict_feature] + ignore_columns)
    y = df[predict_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logging.info(
        "Data processed. Total rows: %d, Train rows: %d, Test rows: %d",
        total_rows,
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test, total_rows


def scale_and_decompose(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": RandomForestRegressor(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
        },
        "Lasso": {
            "alpha": [0.1, 1, 10],
            "max_iter": [1000, 5000],
        },
        "Ridge": {
            "alpha": [0.1, 1, 10],
            "max_iter": [1000, 5000],
        },
    }

    best_models = {}
    all_metrics = []

    for model_name, model in models.items():
        logging.info("Training %s model", model_name)
        grid_search = GridSearchCV(
            model, param_grids[model_name], cv=5, scoring=make_scorer(r2_score)
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "model": model_name,
            "r2_score": r2,
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
        }

        logging.info(
            "%s trained. R2: %.2f, MAE: %.2f, MSE: %.2f", model_name, r2, mae, mse
        )
        logging.info("Best model: %s", best_model)

        all_metrics.append(metrics)
        best_models[model_name] = best_model

    return all_metrics, best_models


def main():
    ensure_directories()

    X_train, X_test, y_train, y_test, total_rows = process_data(
        DATA_PATH, PREDICT_FEATURE, IGNORE_COLUMNS
    )
    feature_names = X_train.columns.tolist()
    X_train_scaled, X_test_scaled, scaler = scale_and_decompose(X_train, X_test)

    metrics, models = train_and_evaluate_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    save_metrics(MODEL_METRICS_SAVE_PATH, metrics)
    save_scaler(SCALER_SAVE_PATH, scaler)

    for model_name, model in models.items():
        model_path = os.path.join(MODELS_SAVE_PATH, f"{model_name}_model.pkl")
        save_model(model_path, model)


if __name__ == "__main__":
    main()
