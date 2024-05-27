# mtb IA

mtb IA is a repository dedicated to predicting route completion times for MTB (Mountain Bike) activities. This
repository contains scripts to process activity data, train a prediction model, and evaluate its performance.

## Repository Structure

- `app/fetch_data.py`: Script to fetch and process wellness and activity data.
- `app/route_time/train_route_time.py`: Script to train a prediction model and evaluate its performance.
- `app/route_time/predict_route_time.py`: Script to make predictions using the trained model.
- `app/Home.py`: Streamlit app script to provide an interactive interface for the project.

## Files

### app/fetch_data.py

This script is responsible for fetching wellness and MTB activity data from an external source and processing it for
later use in model training.

#### Features

- `fetch_wellness(icu, start_date, end_date)`: Fetches wellness data between the specified dates.
- `process_wellness_data(df)`: Processes wellness data, filling missing values and calculating additional metrics.
- `retrieve_activity_data(icu, activity_id)`: Retrieves activity data in FIT format and converts it into a pandas
  DataFrame.

#### Usage

1. Configure the necessary environment variables in the `env.py` file.
2. Run the script to fetch and process the data:
   ```sh
   python app/fetch_data.py
   ```

### app/route_time/train_route_time.py

This script is used to train a route completion time prediction model based on the processed data. It uses
a `RandomForestRegressor` from Scikit-Learn and saves the model metrics to a CSV file.

#### Features

- `process_data(data_path, predict_feature, ignore_columns)`: Processes the data, splitting it into training and testing
  sets.
- `scale_and_decompose(X_train, X_test)`: Scales the data using `StandardScaler`.
- `train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_names)`: Trains the model, evaluates its
  performance, and saves the feature importances.
- `save_metrics(path, metrics)`: Saves the model metrics to a CSV file.

#### Usage

1. Ensure you have the activity data in `app/data/activity_data_summarized.csv`.
2. Run the script to train the model and save the metrics:
   ```sh
   python app/route_time/train_route_time.py
   ```

### app/route_time/predict_route_time.py

This script uses the trained model to make predictions on new data.

#### Usage

1. Ensure the trained model and scaler files are present in the `app/models` directory.
2. Run the script to make predictions:
   ```sh
   python app/route_time/predict_route_time.py
   ```

### app/Home.py

This is the main script for the Streamlit application, providing an interactive interface for the project.

#### Usage

1. Ensure you have all the necessary data and models in place.
2. Run the Streamlit app:
   ```sh
   streamlit run app/Home.py
   ```

## Requirements

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, fitparse, logging, streamlit

## Environment Setup

1. Install the dependencies:
   ```sh
   pip install -r app/requirements.txt
   ```
2. Configure the environment variables in an `env.py` file (see `env.py.template` for an example).

## Contributions

Contributions are welcome. Please open an issue or a pull request to discuss any major changes.

## License

This project is licensed under the MIT License.
