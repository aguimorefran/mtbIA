import gpxpy
import haversine as hs
import joblib
import numpy as np
import pandas as pd
from haversine import Unit

COLS_TO_KEEP = [
    "id",
    "timestamp",
    "position_lat",
    "position_long",
    "distance",
    "altitude"
]

SLOPE_CUTS = [-np.inf, -1, 4, 8, 12, 20, np.inf]
SLOPE_LABELS = ["downhill", "green", "yellow", "orange", "red", "black"]

YEAR_SEASONS = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer",
                7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall", 12: "winter"}

TIME_OF_DAY_CUTS = [0, 6, 12, 18, 24]
TIME_OF_DAY_LABELS = ["night", "morning", "afternoon", "evening"]

WELLNESS_COLS = [
    "atl",
    "ctl",
    "wattkilo"
]

model_path = "web/src/models/lasso_model.pkl"
model_stats_path = "web/src/models/lasso_stats.csv"


def load_model():
    model = joblib.load(model_path)

    stats = pd.read_csv(model_stats_path)
    stats = stats.loc[stats['timestamp'] == stats['timestamp'].max()]
    stats = stats[['r2', 'mae', 'mse']]
    stats = stats.round(3)

    return model, stats


def ingest_gpx(gpx_path):
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file, version='1.1')

    waypoints = [[point.latitude, point.longitude, point.elevation]
                 for track in gpx.tracks
                 for segment in track.segments
                 for point in segment.points]

    data = pd.DataFrame(waypoints, columns=["position_lat", "position_long", "altitude"])

    return data


def process_gpx(data, season, time_of_day, watt_kilo, atl, ctl):
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


def preprocess_gpx(gpx_path, season, time_of_day, watt_kilo, atl, ctl):
    data = ingest_gpx(gpx_path)
    route_agg, route_df = process_gpx(data, season, time_of_day, watt_kilo, atl, ctl)

    return route_agg, route_df


def model_predict(model, route_data):
    # Set season and time of day to lowercase
    route_data["season"] = route_data["season"].str.lower()
    route_data["time_of_day"] = route_data["time_of_day"].str.lower()
    pred_segs = model.predict(route_data)[0]
    hours = pred_segs // 3600
    minutes = (pred_segs % 3600) // 60

    return int(hours), int(minutes)


def predict(model, route_agg, route_df, season, time_of_day, watt_kilo, atl, ctl):
    quarters = [(0.25, 'Quarter 1'), (0.50, 'Quarter 2'), (0.75, 'Quarter 3')]
    quarter_info = []

    for ratio, quarter in quarters:
        quarter_df = route_df[route_df["cum_distance"] <= route_df["cum_distance"].max() * ratio]
        route_agg_qtr, _ = process_gpx(quarter_df, season, time_of_day, watt_kilo, atl, ctl)
        hours, mins = model_predict(model, route_agg_qtr)
        quarter_info.append({
            "Quarter": quarter,
            "position_lat": quarter_df["position_lat"].iloc[-1],
            "position_long": quarter_df["position_long"].iloc[-1],
            "time_str": f"{hours}h {mins}m",
            "distance": quarter_df["cum_distance"].iloc[-1]
        })

    full_h, full_mins = model_predict(model, route_agg)
    quarter_info.append({
        "Quarter": "Full",
        "position_lat": route_df["position_lat"].iloc[-1],
        "position_long": route_df["position_long"].iloc[-1],
        "time_str": f"{full_h}h {full_mins}m",
        "distance": route_df["cum_distance"].iloc[-1]
    })

    return pd.DataFrame(quarter_info)


def plot_map(route_df, prediction):
    import folium

    m = folium.Map(location=[route_df["position_lat"].mean(), route_df["position_long"].mean()], zoom_start=12)

    for idx, row in route_df.iterrows():
        folium.CircleMarker(
            location=[row["position_lat"], row["position_long"]],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue"
        ).add_to(m)

    for idx, row in prediction.iterrows():
        folium.Marker(
            location=[row["position_lat"], row["position_long"]],
            popup=f"Time: {row['time_str']} - Distance: {row['distance']}",
            icon=folium.Icon(color="red")
        ).add_to(m)

    return m
