import os
import pickle

import gpxpy
import haversine as hs
import pandas as pd
from haversine import Unit

from preprocess import SLOPE_CUTS, SLOPE_LABELS

def ingest_gpx(gpx_path):
    """
    Ingest a GPX file and return a DataFrame with the relevant columns
    :param gpx_path: str, path to the GPX file
    :return: pd.DataFrame

    :rtype: pd.DataFrame
    """
    gpx_file = open(gpx_path, 'r')
    gpx = gpxpy.parse(gpx_file)

    waypoints = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                waypoints.append([point.latitude, point.longitude, point.elevation, point.time])

    data = pd.DataFrame(waypoints, columns=["latitude", "longitude", "elevation", "time"])
    data.rename(columns={"latitude": "position_lat", "longitude": "position_long", "elevation": "altitude",
                         "time": "timestamp"}, inplace=True)
    data.drop("timestamp", axis=1, inplace=True)

    return data


def process_gpx(data, season, time_of_day):
    """
    Process the GPX data to calculate the distance, slope, and altitude difference
    :param data: pd.DataFrame, the GPX data
    :param season: str, the season of the activity
    :param time_of_day: str, the time of day of the activity
    :return: pd.DataFrame

    :rtype: pd.DataFrame
    """
    for i in range(1, len(data)):
        data.loc[i, "distance"] = hs.haversine((data.loc[i - 1, "position_lat"], data.loc[i - 1, "position_long"]),
                                               (data.loc[i, "position_lat"], data.loc[i, "position_long"]),
                                               unit=Unit.METERS)
        data["slope"] = (data["altitude"] - data["altitude"].shift(1)) / data["distance"] * 100
        data["altitude_diff"] = data["altitude"].diff()

    # Aggregate the dataframe
    distance = data["distance"].sum()
    ascent_meters = data["altitude_diff"][data["altitude_diff"] > 0].sum()

    # Calculate the percentage of each slope color in the activity per total distance
    data["slope_color"] = pd.cut(data["slope"], bins=SLOPE_CUTS, labels=SLOPE_LABELS)
    data["distance_pctg"] = data["distance"] / data["distance"].sum()

    slopes_pctg = data.groupby("slope_color", observed=False).agg(
        distance_pctg=("distance_pctg", "sum")
    ).reset_index()

    data_dict = {
        "distance": distance,
        "ascent_meters": ascent_meters,
        "season": season,
        "time_of_day": time_of_day
    }

    for idx, row in slopes_pctg.iterrows():
        data_dict[f"{row['slope_color']}_pctg"] = row["distance_pctg"]

    return pd.DataFrame(data_dict, index=[0])


def search_models(models_folder="model_stats", n_models=2, verbose=False):
    """
    Search the models in the models_folder and return the top n_models by R2
    :param models_folder: str, the folder where the models are stored
    :param n_models: int, the number of models to return
    :param verbose: bool, whether to print the models
    :return: pd.DataFrame

    :rtype: pd.DataFrame
    """
    models = {}
    for file in os.listdir(models_folder):
        if file.endswith("_model.pkl"):
            model_name = file.split("_model.pkl")[0]
            models[model_name] = file

    stats = {}
    for model_name, model_file in models.items():
        stats_file = model_file.replace("_model.pkl", "_stats.csv")
        stats[model_name] = stats_file
        # Read the stats file
        stats_df = pd.read_csv(os.path.join(models_folder, stats_file))
        stats[model_name] = stats_df

    # Aggregate the stats in a single dataframe with model_name, and stats, and file_loc for the pkl
    aggregate_stats = pd.DataFrame()
    for model_name, stats_df in stats.items():
        stats_df["model_name"] = model_name
        stats_df["model_file"] = models[model_name]
        aggregate_stats = pd.concat([aggregate_stats, stats_df])

    # Return the top n_models by R2
    top_models = aggregate_stats.sort_values("r2", ascending=False).head(n_models)
    if verbose:
        for idx, row in top_models.iterrows():
            print(f"Model: {row['model_name']}, R2: {row['test_r2']:.2f}")

    return top_models


def make_prediction(model_pkl_path, route_data, model_name):
    """
    Make a prediction with the model
    :param model_pkl_path: str, path to the model pkl file
    :param route_data: pd.DataFrame, the data of the route
    :param model_name: str, the name of the model
    :return: None

    :rtype: None
    """
    # Load model
    model = pickle.load(open(model_pkl_path, 'rb'))
    pred_segs = model.predict(route_data)[0]
    hours = pred_segs // 3600
    minutes = (pred_segs % 3600) // 60

    print(f"Predicted time for {model_name}: {int(hours)} hours and {int(minutes)} minutes")

### MAIN SCRIPT ###

SEASON = "summer"
TIME_OF_DAY = "morning"

gpx_folder = "../data/gpx"
preprocessor_pkl_path = "preprocessor.pkl"

available_models = search_models(n_models=2)

for file in os.listdir(gpx_folder):
    if file.endswith(".gpx"):
        gpx_path = os.path.join(gpx_folder, file)
        df = ingest_gpx(gpx_path)
        print("-" * 50)
        print(f"Processing {gpx_path}")
    route_data = process_gpx(df, SEASON, TIME_OF_DAY)

    for idx, model_info in available_models.iterrows():
        model_pkl_path = os.path.join("model_stats", model_info["model_file"])
        try:
            make_prediction(model_pkl_path, route_data, model_info["model_name"])
        except Exception as e:
            print(f"Error predicting with model {model_info['model_name']}: {e}")
            continue