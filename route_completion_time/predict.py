import os
import pickle

import contextily as ctx
import gpxpy
import haversine as hs
import pandas as pd
import xyzservices as xzy
from haversine import Unit
from matplotlib import pyplot as plt

from preprocess import SLOPE_CUTS, SLOPE_LABELS


def ingest_gpx(gpx_path):
    """
    Ingest the GPX file and return a DataFrame with the waypoints
    :param gpx_path: str, the path to the GPX file
    :return: pd.DataFrame
    """
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file, version='1.1')

    waypoints = [[point.latitude, point.longitude, point.elevation]
                 for track in gpx.tracks
                 for segment in track.segments
                 for point in segment.points]

    data = pd.DataFrame(waypoints, columns=["position_lat", "position_long", "altitude"])

    return data


def process_gpx(data, season, time_of_day, watt_kilo, atl, ctl):
    """
    Process the GPX data to calculate the distance, slope, and altitude difference. Return a DataFrame with the
    aggregated data and the raw data
    :param data: pd.DataFrame, the GPX data
    :param season: str, the season of the activity
    :param time_of_day: str, the time of day of the activity
    :param watt_kilo: float, the watt per kilogram of the athlete
    :param atl: int, the acute training load of the athlete
    :param ctl: int, the chronic training load of the athlete
    :return: tuple(pd.DataFrame, pd.DataFrame)

    :rtype: pd.DataFrame
    """

    # Reset idx
    data = data.copy()
    data.reset_index(drop=True, inplace=True)

    for i in range(1, len(data)):
        data.loc[i, "distance"] = hs.haversine((data.loc[i - 1, "position_lat"], data.loc[i - 1, "position_long"]),
                                               (data.loc[i, "position_lat"], data.loc[i, "position_long"]),
                                               unit=Unit.METERS)
        data["slope"] = (data["altitude"] - data["altitude"].shift(1)) / data["distance"] * 100
        data["altitude_diff"] = data["altitude"].diff()

    data["cum_distance"] = data["distance"].cumsum()

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

    return pd.DataFrame(data_dict, index=[0]), data


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


def plot_prediction(gpx_data, pred_time_hours, pred_time_minutes, route_name, distance, ascent, quarter_info):
    """
    Plot the GPX data with predicted time and colored slope
    :param gpx_data: pd.DataFrame, the GPX data with required columns
    :param pred_time_hours: int, predicted time in hours
    :param pred_time_minutes: int, predicted time in minutes
    :param route_name: str, name of the route
    :param distance: float, distance of the route
    :param ascent: float, ascent of the route
    :param quarter_info: pd.DataFrame, the info of the quarters
    :return: None

    :rtype: None
    """
    gpx_data = gpx_data.copy()
    pred_time_hours = int(pred_time_hours)
    pred_time_minutes = int(pred_time_minutes)
    formatted_time = f"{pred_time_hours}h {pred_time_minutes}m"

    gpx_data['slope_color'] = gpx_data['slope_color'].fillna("downhill").map(lambda x: "gray" if x == "downhill" else x)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(gpx_data) - 1):
        ax.plot([gpx_data['position_long'][i], gpx_data['position_long'][i + 1]],
                [gpx_data['position_lat'][i], gpx_data['position_lat'][i + 1]],
                color=gpx_data['slope_color'].iloc[i], linewidth=2)


    ax.plot(gpx_data['position_long'].iloc[0], gpx_data['position_lat'].iloc[0], '*', color='blue', label='Start')

    # Start point text
    start_lat = gpx_data['position_lat'].iloc[0]
    start_long = gpx_data['position_long'].iloc[0]
    ax.text(start_long, start_lat, 'Start\n' + formatted_time, color='black', style='italic', fontsize=12, weight='bold')

    # Add background map using contextily
    ctx.add_basemap(ax, crs="EPSG:4326",
                    source=xzy.providers.OpenStreetMap.Mapnik)

    # Add quarter points with an asterisk, and the time inside a box
    for idx, row in quarter_info.iterrows():
        ax.plot(row["position_long"], row["position_lat"], '*', color='red', label='Quarter')
        ax.text(row["position_long"], row["position_lat"], row["time_str"], color='black', style='italic', fontsize=12,
                weight='bold')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    plt.title(
        "Route: " + route_name + f"\nDistance: {distance:.2f} km, Ascent: {ascent:.2f} meters" + "\nPredicted time: " + str(
            formatted_time))

    # Save plot
    plt.savefig(f"model_stats/{route_name}_prediction.png", bbox_inches='tight')

    return None


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

    # print(f"Predicted time for {model_name}: {int(hours)} hours and {int(minutes)} minutes")

    return int(hours), int(minutes)


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

# RACE_ATL = 51
# RACE_CTL = 31


gpx_folder = "../data/gpx"
preprocessor_pkl_path = "preprocessor.pkl"

available_models = search_models(n_models=n_models)

for file in os.listdir(gpx_folder):
    if file.endswith(".gpx"):
        # Ask user to continue with this file
        gpx_path = os.path.join(gpx_folder, file)
        df = ingest_gpx(gpx_path)
        print("-" * 50)
        print(f"Processing {gpx_path}")
        print("Do you want to skip this file? (y/n)")
        skip = input()
        if skip == "y":
            continue

        route_agg, route_df = process_gpx(df, SEASON, TIME_OF_DAY, WATTKILO, ATL, CTL)
        distance_km = route_agg["distance"].values[0] / 1000
        ascent_meters = route_agg["ascent_meters"].values[0]

        # Separate the route in 3 chunks: 0-25%, 0-50%, 0-75%
        first_qtr = route_df[route_df["cum_distance"] <= route_df["cum_distance"].max() * 0.25]
        second_qtr = route_df[route_df["cum_distance"] <= route_df["cum_distance"].max() * 0.50]
        third_qtr = route_df[route_df["cum_distance"] <= route_df["cum_distance"].max() * 0.75]

        route_agg_1, _ = process_gpx(first_qtr, SEASON, TIME_OF_DAY, WATTKILO, ATL, CTL)
        route_agg_2, _ = process_gpx(second_qtr, SEASON, TIME_OF_DAY, WATTKILO, ATL, CTL)
        route_agg_3, _ = process_gpx(third_qtr, SEASON, TIME_OF_DAY, WATTKILO, ATL, CTL)

        for idx, model_info in available_models.iterrows():
            model_pkl_path = os.path.join("model_stats", model_info["model_file"])
            print(f"Making prediction with {model_info['model_name']}")

            full_h, full_mins = make_prediction(model_pkl_path, route_agg, model_info["model_name"])
            first_h, first_mins = make_prediction(model_pkl_path, route_agg_1, model_info["model_name"])
            second_h, second_mins = make_prediction(model_pkl_path, route_agg_2, model_info["model_name"])
            third_h, third_mins = make_prediction(model_pkl_path, route_agg_3, model_info["model_name"])

            # Create a dict with the quarter info, time str, lat, long
            quarter_info = pd.DataFrame({
                "position_lat": [first_qtr["position_lat"].iloc[-1], second_qtr["position_lat"].iloc[-1],
                                 third_qtr["position_lat"].iloc[-1]],
                "position_long": [first_qtr["position_long"].iloc[-1], second_qtr["position_long"].iloc[-1],
                                  third_qtr["position_long"].iloc[-1]],
                "time_str": [f"{first_h}h {first_mins}m", f"{second_h}h {second_mins}m", f"{third_h}h {third_mins}m"]
            })

            print(f"First quarter: {first_h} hours and {first_mins} minutes")
            print(f"Second quarter: {second_h} hours and {second_mins} minutes")
            print(f"Third quarter: {third_h} hours and {third_mins} minutes")
            print(f"Full route: {full_h} hours and {full_mins} minutes")

            print("Plotting map")
            plot_prediction(route_df, full_h, full_mins, file, distance_km,
                            ascent_meters, quarter_info)
