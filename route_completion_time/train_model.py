import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR


def save_model(model, model_path, stats_path, stat_dict):
    # Crete folder model_stats if it doesn't exist
    model_path = f"model_stats/{model_path}"
    stats_path = f"model_stats/{stats_path}"

    # Create folder model_stats if it doesn't exist
    if not os.path.exists("model_stats"):
        os.makedirs("model_stats")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    pd.DataFrame([stat_dict]).to_csv(stats_path, index=False)
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {stats_path}")


def prepare_data(data_df):
    target_feature = "elapsed_time"
    X = data_df.drop(target_feature, axis=1)
    y = data_df[target_feature]

    cat_features = X.select_dtypes(include=["object"]).columns
    num_features = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(), cat_features)
        ])


    return X, y, preprocessor
def train_ridge_model(data_df, save_model_path, save_model_stats_path):
    print("-" * 50)
    print("Training Ridge model")
    X, y, preprocessor = prepare_data(data_df)

    ridge = Ridge()

    param_grid = {
        'ridge__alpha': [0.001, 0.1, 1, 10, 100],
        'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'ridge__max_iter': [1000, 5000, 10000, 50000]
    }

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ridge', ridge)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(pipeline, param_grid, cv=4, scoring='neg_mean_squared_error', return_train_score=True,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    print("Finished training Ridge model")


    final_model = grid_search.best_estimator_
    predictions = final_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Best params: ", grid_search.best_params_)
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2: ", r2)

    stat_dict = {
        'mse': mse, 'mae': mae, 'r2': r2,
        'best_params': grid_search.best_params_
    }
    save_model(final_model, save_model_path, save_model_stats_path, stat_dict)


data_df = pd.read_csv("preprocessed.csv")
data_df.drop(data_df.columns[[0, 1]], axis=1, inplace=True)
data_df.dropna(inplace=True)


def train_svr_lineal_model(data_df, save_model_path, save_model_stats_path):
    print("-" * 50)
    print("Training SVR Linear model")
    X, y, preprocessor = prepare_data(data_df)



    svr = SVR(kernel='linear')

    param_grid = {
        'svr__C': [0.001, 0.1, 1, 10, 100],
        'svr__epsilon': [0.1, 0.2, 0.5, 1],
        'svr__max_iter': [1000, 5000, 10000, 50000]
    }

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('svr', svr)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(pipeline, param_grid, cv=4, scoring='neg_mean_squared_error', return_train_score=True,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    print("Finished training SVR Linear model")

    final_model = grid_search.best_estimator_
    predictions = final_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Best params: ", grid_search.best_params_)
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2: ", r2)

    stat_dict = {
        'mse': mse, 'mae': mae, 'r2': r2,
        'best_params': grid_search.best_params_
    }

    save_model(final_model, save_model_path, save_model_stats_path, stat_dict)

    hyperparams = {
        'hidden_layer_sizes': (15, 15, 15, 10),
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate': 'adaptive',
        'learning_rate_init': 1e-5,
        'alpha': 0.001,
        'max_iter': 100000
    }

    # Prepared data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(preprocessor, MLPRegressor(**hyperparams))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Finished training MLP model")
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2: ", r2)


train_ridge_model(data_df, "ridge_model.pkl", "ridge_stats.csv")
train_svr_lineal_model(data_df, "svr_model.pkl", "svr_stats.csv")
