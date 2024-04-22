import os

import gpxpy
import haversine as hs
import pandas as pd
import pickle
from haversine import Unit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

from preprocess import SLOPE_CUTS, SLOPE_LABELS

gpx_path = "../data/gpx/ruta_balcon_axarquia.gpx"
preprocessor_pkl_path = "preprocessor.pkl"


def ingest_gpx(gpx_path):
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

def search_models(models_folder="model_stats"):
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


    return aggregate_stats

def preprocess_data(data_df, preprocessor_pkl_path="preprocessor.pkl"):
    preprocessor = pickle.load(open(preprocessor_pkl_path, 'rb'))
    return preprocessor.transform(data_df)


def make_prediction(model_pkl_path, route_data, model_name):
    # Load model
    model = pickle.load(open(model_pkl_path, 'rb'))
    pred_minutes = model.predict(route_data)
    print(f"Predicted time for {model_name}: {pred_minutes[0]:.2f} minutes")


SEASON = "summer"
TIME_OF_DAY = "morning"

df = ingest_gpx(gpx_path)
route_data = process_gpx(df, SEASON, TIME_OF_DAY)
available_models = search_models()
print(available_models)

# # Select first model
# model_info = available_models.iloc[0]
# model_pkl_path = os.path.join("model_stats", model_info["model_file"])
# make_prediction(model_pkl_path, route_data)

# Predict with every model
for idx, model_info in available_models.iterrows():
    model_pkl_path = os.path.join("model_stats", model_info["model_file"])
    make_prediction(model_pkl_path, route_data, model_info["model_name"])

