import os
import pickle

import gpxpy
import haversine as hs
import pandas as pd
from haversine import Unit
from matplotlib import pyplot as plt
import xyzservices as xzy

from preprocess import SLOPE_CUTS, SLOPE_LABELS


def ingest_gpx(gpx_path):
    """
    Ingest a GPX file and return a DataFrame with the relevant columns
    :param gpx_path: str, path to the GPX file
    :return: pd.DataFrame

    :rtype: pd.DataFrame
    """
    gpx_file = open(gpx_path, 'r')
    gpx = gpxpy.parse(gpx_file, version='1.1')

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


def process_gpx(data, season, time_of_day, watt_kilo, atl, ctl):
    """
    Process the GPX data to calculate the distance, slope, and altitude difference
    :param data: pd.DataFrame, the GPX data
    :param season: str, the season of the activity
    :param time_of_day: str, the time of day of the activity
    :param watt_kilo: float, the watt per kilogram of the athlete
    :param atl: int, the acute training load of the athlete
    :param ctl: int, the chronic training load of the athlete
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
        "time_of_day": time_of_day,
        "wattkilo": watt_kilo,
        "atl": atl,
        "ctl": ctl
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


def plot_prediction(gpx_data, pred_time_hours, pred_time_minutes, route_name, distance, ascent):
    """
    Plot the GPX data with predicted time and colored slope
    :param gpx_data: pd.DataFrame, the GPX data with required columns
    :param pred_time_hours: int, predicted time in hours
    :param pred_time_minutes: int, predicted time in minutes
    :param route_name: str, name of the route
    :param distance: float, distance of the route
    :param ascent: float, ascent of the route
    :return: None

    :rtype: None
    """
    gpx_data = gpx_data.copy()
    pred_time_hours = int(pred_time_hours)
    pred_time_minutes = int(pred_time_minutes)
    formatted_time = f"{pred_time_hours}h {pred_time_minutes}m"

    gpx_data['slope_color'] = gpx_data['slope_color'].fillna("downhill").map(lambda x: "gray" if x == "downhill" else x)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot GPX data with colored slope
    for i in range(len(gpx_data) - 1):
        ax.plot([gpx_data['position_long'][i], gpx_data['position_long'][i + 1]],
                [gpx_data['position_lat'][i], gpx_data['position_lat'][i + 1]],
                color=gpx_data['slope_color'].iloc[i], linewidth=2)

    # Add start point to the plot as a blue star
    ax.plot(gpx_data['position_long'].iloc[0], gpx_data['position_lat'].iloc[0], '*', color='blue', label='Start')

    # Add background map using contextily
    ctx.add_basemap(ax, crs="EPSG:4326",
                    source=xzy.providers.OpenStreetMap.Mapnik)

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    plt.title("Route: " + route_name + f"\nDistance: {distance:.2f} km, Ascent: {ascent:.2f} meters" + "\nPredicted time: " + str(formatted_time))

    # Save plot
    plt.savefig(f"model_stats/{route_name}_prediction.png", bbox_inches='tight')


def predict_10km_time(watt_kilo, atl, ctl, season, time_of_day, gpx_path, model_pkl_path, model_name):
    # Receive the entire GPX file
    # Traverse the GPX file and calculate the cumsum distance
    # Every 10km calculate the time with the model for the traveled distance
    # Save in a new df the coords, distance and predicted time

    predictions = pd.DataFrame(columns=["latitude", "longitude", "distance", "predicted_time"])
    gpx_data = ingest_gpx(gpx_path)
    distance = 0
    for i in range(1, len(gpx_data)):
        distance += hs.haversine((gpx_data.loc[i - 1, "position_lat"], gpx_data.loc[i - 1, "position_long"]),
                                 (gpx_data.loc[i, "position_lat"], gpx_data.loc[i, "position_long"]),
                                 unit=Unit.METERS)
        if distance >= 10000:
            route_data = process_gpx(gpx_data[:i], season, time_of_day, watt_kilo, atl, ctl)
            hours, minutes = make_prediction(model_pkl_path, route_data, model_name)
            predictions = predictions.append({"latitude": gpx_data.loc[i, "position_lat"],
                                              "longitude": gpx_data.loc[i, "position_long"],
                                              "distance": distance,
                                              "predicted_time": hours * 3600 + minutes * 60}, ignore_index=True)
            distance = 0

    return predictions


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

    ### 10kms prediction
    predictions_10km = predict_10km_time(route_data["wattkilo"].values[0], route_data["atl"].values[0],
                                         route_data["ctl"].values[0], route_data["season"].values[0],
                                         route_data["time_of_day"].values[0], gpx_path, model_pkl_path, model_name)

    print(predictions_10km)

    return hours, minutes


### MAIN SCRIPT ###
# FITNESS = CTL
# FATIGUE = ATL

n_models = 1
SEASON = "spring"
TIME_OF_DAY = "morning"
WATTS = 220
KILOS = 90
WATTKILO = WATTS / KILOS
ATL = 51
CTL = 31

RACE_ATL = 51
RACE_CTL = 31

print(f"Watt per kilo: {WATTKILO}")

gpx_folder = "../data/gpx"
preprocessor_pkl_path = "preprocessor.pkl"

available_models = search_models(n_models=n_models)

for file in os.listdir(gpx_folder):
    if file.endswith(".gpx"):
        gpx_path = os.path.join(gpx_folder, file)
        df = ingest_gpx(gpx_path)
        print("-" * 50)
        print(f"Processing {gpx_path}")

        route_data = process_gpx(df, SEASON, TIME_OF_DAY, WATTKILO, ATL, CTL)
        distance_km = route_data["distance"].values[0] / 1000
        ascent_meters = route_data["ascent_meters"].values[0]
        print(f"Distance: {distance_km:.2f} km")
        print(f"Ascent: {ascent_meters:.2f} meters")

        for idx, model_info in available_models.iterrows():
            model_pkl_path = os.path.join("model_stats", model_info["model_file"])
            try:
                hours, minutes = make_prediction(model_pkl_path, route_data, model_info["model_name"])

            except Exception as e:
                print(f"Error predicting with model {model_info['model_name']}: {e}")
                continue

            # plot_prediction(df, hours, minutes, file, distance_km,
            #                 ascent_meters)