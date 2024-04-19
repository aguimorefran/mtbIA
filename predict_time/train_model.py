import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle


def oversample(df, row_ratio=0.3, sample_min = 1.05, sample_max=1.15, target_feature="duration_minutes"):
    """
    Oversample the data to create new rows with. Creates new rows with the target feature modified.
    :param df: Dataframe with the data
    :param row_ratio: Ratio of rows to be created
    :param sample_ratio: Ratio space for the new data
    :param target_feature: Target feature to be modified
    :return: Dataframe with the new rows
    :rtype: pd.DataFrame
    """

    n_new_rows = int(len(df) * row_ratio)
    oversampled_df = pd.DataFrame(columns=df.columns)

    for i in range(n_new_rows):
        row = df.sample()
        row = row.copy()
        org_value = row[target_feature].values[0]
        new_value = np.random.uniform(org_value * sample_min, org_value * sample_max)
        row[target_feature] = new_value
        oversampled_df = pd.concat([oversampled_df, row])

    return oversampled_df


def save_model(model, model_path, stats_path, stat_dict, best_model=None):
    """
    Save model and stats to disk in same directory
    :param model: Model to be saved
    :param model_path: Path to save model
    :param stats_path: Path to save stats in CSV
    :param stat_dict: Dictionary with stats
    :param best_model: Best model to be saved
    :return: List with real paths to saved files
    :rtype: List
    """

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save stats
    pd.DataFrame(stat_dict).to_csv(stats_path, index=False)

    if best_model:
        best_model_path = model_path.replace(".pkl", "_best.pkl")
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)

    print("Model saved to: ", model_path)
    print("Stats saved to: ", stats_path)

    return [model_path, stats_path]



data_df = pd.read_csv("preprocessed.csv")

# Remove idx=0 and idx=1 cols
data_df = data_df.drop(data_df.columns[[0, 1]], axis=1)

# Create new dataframe with oversampled data
oversamples = oversample(data_df)

# Concatenate dataframes
# oversampled_df = pd.concat([data_df, oversamples])
#
# print("Dataframe shape: ", data_df.shape)
# print("Oversampled dataframe shape: ", oversampled_df.shape)

# Split data into X and y
target_feature = "duration_minutes"
X = data_df.drop(target_feature, axis=1)
y = data_df[target_feature]


def train_ridge_model(X, y):
    print("--- Training Ridge model ---")
    # Create pipeline
    # OHE categorical features
    # Standardize numerical features

    cat_features = list(X.select_dtypes(include=['object']).columns)
    num_features = list(X.select_dtypes(include=['int64', 'float64']).columns)

    # Split into train and test
    train_set, test_set = train_test_split(data_df, test_size=0.2, random_state=42)

    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = ColumnTransformer([
        ('encoder', OneHotEncoder(), cat_features)
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    data_prepared = full_pipeline.fit_transform(train_set)

    # Train Ridge model
    cv = 4
    scoring = "neg_mean_squared_error"
    ridge = Ridge()
    param_grid = {
        'alpha': [0.001, 0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': [1000, 5000, 10000, 50000]
    }

    def grid_search(param_grid, cv, scoring, verbose):
        return GridSearchCV(ridge, param_grid, cv=cv, scoring=scoring, return_train_score=True, verbose=verbose)

    grid_search = grid_search(param_grid, cv, 'neg_mean_squared_error', 1)
    grid_search.fit(data_prepared, train_set[target_feature])

    # Evaluate model
    final_model = grid_search.best_estimator_
    data_test = full_pipeline.transform(test_set)
    predictions = final_model.predict(data_test)
    mse = mean_squared_error(test_set[target_feature], predictions)
    mae = mean_absolute_error(test_set[target_feature], predictions)
    r2 = r2_score(test_set[target_feature], predictions)

    print("Ridge model trained on original data")
    print("Best params: ", grid_search.best_params_)
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2: ", r2)

    # Save model
    model_path = "ridge_model.pkl"
    stats_path = "ridge_stats.csv"
    stat_dict = {
        "rows": [len(data_df)],
        "columns": [len(data_df.columns)],
        "cv": [cv],
        "scoring": [scoring],
        "mse": [mse],
        "mae": [mae],
        "r2": [r2]
    }
    save_model(final_model, model_path, stats_path, stat_dict)


def train_svr_model(X, y):
    pass


def train_mlp_model(X, y):
    pass


train_ridge_model(X, y)
