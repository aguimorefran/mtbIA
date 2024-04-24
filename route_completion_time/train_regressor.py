import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR

def save_model(model, model_path, stats_path, stat_dict, col_names):
    # Create folder model_stats if it doesn't exist
    model_path = f"model_stats/{model_path}"
    stats_path = f"model_stats/{stats_path}"

    if not os.path.exists("model_stats"):
        os.makedirs("model_stats")

    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Add timestamp and n_cols to the stats
    stat_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stat_dict['col_names'] = col_names

    # If stats file exists, append a new row. Otherwise, create a new file.
    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        df = pd.concat([df, pd.DataFrame([stat_dict])])
    else:
        df = pd.DataFrame([stat_dict])

    df.to_csv(stats_path, index=False)
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {stats_path}")


def prepare_data(data_df, target_feature):
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



def train_regressor(data_df, save_model_path, save_model_stats_path, params, regressor_object, regressor_name, target_feature):
    print("-" * 50)
    print(f"Training {regressor_name} model")

    X, y, preprocessor = prepare_data(data_df, target_feature)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', regressor_object)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(pipeline, params, cv=4, scoring='neg_mean_squared_error', return_train_score=True,
                               verbose=1)

    grid_search.fit(X_train, y_train)

    print(f"Finished training {regressor_name} model")

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

    save_model(final_model, save_model_path, save_model_stats_path, stat_dict, data_df.columns)


def train_MLP(data_df, save_model_path, save_model_stats_path, regressor_name, target_feature):
    print("-" * 50)
    print(f"Training {regressor_name} model")

    X, y, preprocessor = prepare_data(data_df, target_feature)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPRegressor()
    pipeline = make_pipeline(preprocessor, model)

    X_processed = preprocessor.fit_transform(X_train)
    print("Number of input features: ", X_processed.shape[1])


    space = {
        'hidden_layer_sizes':
            hp.choice('hidden_layer_sizes', [
                32, 50,
                (20, 20),
                (50, 50),
                (50, 50, 10),
            ]),
        'activation': hp.choice('activation', ['relu', 'tanh']),
        'learning_rate_init': hp.loguniform(
            'learning_rate_init', np.log(1e-5), np.log(1e-2)),
        'max_iter': 50000,
        'early_stopping': True,
        'solver': ['adam', 'lbfgs']
    }

    def objective(params):
        print(params)
        pipeline.set_params(mlpregressor__hidden_layer_sizes=params['hidden_layer_sizes'],
                            mlpregressor__activation=params['activation'],
                            mlpregressor__learning_rate_init=params['learning_rate_init'],
                            mlpregressor__early_stopping=params['early_stopping'],
                            mlpregressor__solver=params['solver'],
                            mlpregressor__max_iter=params['max_iter'])
        score = -cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    best_params = {
        'hidden_layer_sizes': space['hidden_layer_sizes'][best['hidden_layer_sizes']],
        'activation': space['activation'][best['activation']],
        'learning_rate_init': best['learning_rate_init'],
        'early_stopping': space['early_stopping'],
        'solver': space['solver'],
        'max_iter': space['max_iter']
    }

    print("Best params: ", best_params)

    pipeline.set_params(mlpregressor__hidden_layer_sizes=best_params['hidden_layer_sizes'],
                        mlpregressor__activation=best_params['activation'],
                        mlpregressor__learning_rate_init=best_params['learning_rate_init'],
                        mlpregressor__early_stopping=best_params['early_stopping'],
                        mlpregressor__solver=best_params['solver'],
                        mlpregressor__max_iter=best_params['max_iter'])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2: ", r2)


data_df = pd.read_csv("preprocessed.csv")
data_df.drop(data_df.columns[[0, 1]], axis=1, inplace=True)
data_df.dropna(inplace=True)
target_feature = "elapsed_time"


train_MLP(data_df, "mlp_model.pkl", "mlp_stats.csv", "MLP", target_feature)

################
# Ridge Model #
################
ridge_reg = Ridge()
ridge_reg_params = {
    'regressor__alpha': [0.001, 0.1, 1, 10, 100],
    'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    'regressor__max_iter': [1000, 5000, 10000, 50000]
}
ridge_reg_name = "Ridge"
ridge_model_path = "ridge_model.pkl"
ridge_stats_path = "ridge_stats.csv"

################
# SVR Linear Model #
################

svr_lin_reg = SVR(kernel='linear')
svr_lin_reg_params = {
    'regressor__C': [0.001, 0.1, 1, 10, 100],
    'regressor__epsilon': [0.1, 0.2, 0.5, 1],
    'regressor__max_iter': [1000, 5000, 10000, 50000]
}
svr_lin_reg_name = "SVR Linear"
svr_lin_model_path = "svr_lin_model.pkl"
svr_lin_stats_path = "svr_lin_stats.csv"

################
# SVR RBF Model #
################

svr_rbf_reg = SVR(kernel='rbf')
svr_rbf_reg_params = {
    'regressor__C': [0.001, 0.1, 1, 10, 100],
    'regressor__epsilon': [0.1, 0.2, 0.5, 1],
    'regressor__gamma': ['scale', 'auto']
}
svr_rbf_reg_name = "SVR RBF"
svr_rbf_model_path = "svr_rbf_model.pkl"
svr_rbf_stats_path = "svr_rbf_stats.csv"

################
# Lasso Model #
################

lasso_reg = Lasso()
lasso_reg_params = {
    'regressor__alpha': [0.001, 0.1, 1, 10, 100],
    'regressor__max_iter': [1000, 5000, 10000, 50000]
}
lasso_reg_name = "Lasso"
lasso_model_path = "lasso_model.pkl"
lasso_stats_path = "lasso_stats.csv"

################
# ElasticNet Model #
################

elastic_net_reg = ElasticNet()
elastic_net_reg_params = {
    'regressor__alpha': [0.001, 0.1, 1, 10, 100],
    'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9],
    'regressor__max_iter': [10000000]
}
elastic_net_reg_name = "ElasticNet"
elastic_net_model_path = "elastic_net_model.pkl"
elastic_net_stats_path = "elastic_net_stats.csv"

# train_regressor(data_df, svr_lin_model_path, svr_lin_stats_path, svr_lin_reg_params, svr_lin_reg, svr_lin_reg_name, target_feature)
# train_regressor(data_df, ridge_model_path, ridge_stats_path, ridge_reg_params, ridge_reg, ridge_reg_name, target_feature)
# # train_regressor(data_df, svr_rbf_model_path, svr_rbf_stats_path, svr_rbf_reg_params, svr_rbf_reg, svr_rbf_reg_name, target_feature)
# train_regressor(data_df, lasso_model_path, lasso_stats_path, lasso_reg_params, lasso_reg, lasso_reg_name, target_feature)
# train_regressor(data_df, elastic_net_model_path, elastic_net_stats_path, elastic_net_reg_params, elastic_net_reg,
#                 elastic_net_reg_name, target_feature)