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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/activity_data.csv"
MODEL_METRICS_SAVE_PATH = "data/model_metrics.csv"
MODEL_SAVE_PATH = "models/route_completion_time.pkl"

PREDICT_FEATURE = "duration_seconds"
IGNORE_COLUMNS = ["activity_id", "date"]
PCA_VAR = 0.99


def ensure_directories():
    os.makedirs(os.path.dirname(MODEL_METRICS_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


def save_metrics(path, metrics):
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    df.to_csv(path, index=False)
    logging.info("Metrics saved successfully.")


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
    return X_train_scaled, X_test_scaled


def train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_names):
    model = RandomForestRegressor()
    model_param_grid = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    grid_search = GridSearchCV(
        model, model_param_grid, cv=5, scoring=make_scorer(r2_score)
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Obtener importancias de las características
    feature_importances = best_model.feature_importances_
    feature_importance_dict = {
        name: importance for name, importance in zip(feature_names, feature_importances)
    }

    metrics = {
        "r2_score": r2,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "feature_importances": feature_importance_dict,  # Añadir importancias de características a las métricas
    }
    return metrics, best_model


def main():
    ensure_directories()

    X_train, X_test, y_train, y_test, total_rows = process_data(
        DATA_PATH, PREDICT_FEATURE, IGNORE_COLUMNS
    )
    feature_names = X_train.columns.tolist()
    X_train_scaled, X_test_scaled = scale_and_decompose(X_train, X_test)

    metrics, model = train_and_evaluate_model(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_names
    )

    save_metrics(MODEL_METRICS_SAVE_PATH, metrics)

    with open(MODEL_SAVE_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    logging.info("Model saved successfully.")


if __name__ == "__main__":
    main()
