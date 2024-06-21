import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = "data/activity_data_summarized.csv"
MODEL_METRICS_SAVE_PATH = "data/model_metrics_route_time.csv"
MODELS_SAVE_PATH = "models/"

PREDICT_FEATURE = "duration_seconds"
IGNORE_COLUMNS = ["activity_id", "date"]

def ensure_directories():
    os.makedirs(os.path.dirname(MODEL_METRICS_SAVE_PATH), exist_ok=True)
    os.makedirs(MODELS_SAVE_PATH, exist_ok=True)

def save_metrics(path: str, metrics: List[Dict]):
    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.Timestamp.now()
    df.to_csv(path, index=False)
    logging.info(f"Metrics saved to {path}")

def save_model(path: str, model: object):
    with open(path, "wb") as model_file:
        pickle.dump(model, model_file)
    logging.info(f"Model saved to {path}")

def process_data(data_path: str, predict_feature: str, ignore_columns: List[str]) -> Tuple:
    df = pd.read_csv(data_path).dropna()
    total_rows = df.shape[0]
    X = df.drop(columns=[predict_feature] + ignore_columns)
    y = df[predict_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data processed. Total rows: {total_rows}, Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test, total_rows

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> Tuple[List[Dict], Dict]:
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42),
        "ElasticNet": ElasticNet(random_state=42),
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "Lasso": {
            "alpha": [0.1, 1, 10],
            "max_iter": [1000, 5000],
        },
        "Ridge": {
            "alpha": [0.1, 1, 10],
            "max_iter": [1000, 5000],
        },
        "ElasticNet": {
            "alpha": [0.1, 1, 10],
            "l1_ratio": [0.1, 0.5, 0.9],
            "max_iter": [1000, 5000],
        },
    }

    best_models = {}
    all_metrics = []

    for model_name, model in models.items():
        logging.info(f"Training {model_name} model")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        param_grid = {'model__' + key: value for key, value in param_grids[model_name].items()}
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -cv_scores
        r2_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')

        metrics = {
            "model": model_name,
            "r2_score_mean": np.mean(r2_scores),
            "r2_score_std": np.std(r2_scores),
            "mse_mean": np.mean(mse_scores),
            "mse_std": np.std(mse_scores),
            "mae_mean": np.mean(np.sqrt(mse_scores)),
            "mae_std": np.std(np.sqrt(mse_scores)),
        }

        logging.info(f"{model_name} trained. R2: {metrics['r2_score_mean']:.2f} (+/- {metrics['r2_score_std']:.2f}), "
                     f"MSE: {metrics['mse_mean']:.2f} (+/- {metrics['mse_std']:.2f})")
        logging.info(f"Best parameters: {grid_search.best_params_}")

        all_metrics.append(metrics)
        best_models[model_name] = best_model

    return all_metrics, best_models

def main():
    ensure_directories()

    X_train, X_test, y_train, y_test, total_rows = process_data(DATA_PATH, PREDICT_FEATURE, IGNORE_COLUMNS)
    feature_names = X_train.columns.tolist()

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    metrics, models = train_and_evaluate_model(X, y)

    save_metrics(MODEL_METRICS_SAVE_PATH, metrics)

    for model_name, model in models.items():
        model_path = os.path.join(MODELS_SAVE_PATH, f"{model_name}_model_route_time.pkl")
        save_model(model_path, model)

    return metrics, models, feature_names, total_rows

if __name__ == "__main__":
    main()