import gpxpy
import haversine as hs
import joblib
import numpy as np
import pandas as pd
from haversine import Unit
from streamlit_folium import st_folium
import folium

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


def process_gpx(data, season, time_of_day, watt_kilo, atl, ctl, pbar):
    pbar_min = 20
    pbar_max = 100
    data = data.copy()
    data.reset_index(drop=True, inplace=True)

    for i in range(1, len(data)):
        data.loc[i, "distance"] = hs.haversine((data.loc[i - 1, "position_lat"], data.loc[i - 1, "position_long"]),
                                               (data.loc[i, "position_lat"], data.loc[i, "position_long"]),
                                               unit=Unit.METERS)
        data["slope"] = (data["altitude"] - data["altitude"].shift(1)) / data["distance"] * 100
        data["altitude_diff"] = data["altitude"].diff()
        pb = pbar_min + (pbar_max - pbar_min) * i / len(data)
        pb_pctg = pb / pbar_max * 100
        pbar.progress(int(pb), f"Processing GPX data... ({pb_pctg:.2f}%)")

    pbar.progress(pbar_max, "Done processing GPX data")

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


def preprocess_gpx(gpx_path, season, time_of_day, watt_kilo, atl, ctl, pbar):
    pbar.progress(10, "Ingesting GPX file...")
    data = ingest_gpx(gpx_path)
    pbar.progress(20, "Processing GPX data...")
    route_agg, route_df = process_gpx(data, season, time_of_day, watt_kilo, atl, ctl, pbar)

    return route_agg, route_df


def model_predict(model, route_data):
    # Set season and time of day to lowercase
    route_data["season"] = route_data["season"].str.lower()
    route_data["time_of_day"] = route_data["time_of_day"].str.lower()
    pred_segs = model.predict(route_data)[0]
    hours = pred_segs // 3600
    minutes = (pred_segs % 3600) // 60

    return int(hours), int(minutes)


def predict(model, route_agg, route_df, season, time_of_day, watt_kilo, atl, ctl, pbar):
    quarters = [(0.25, 'Quarter 1'), (0.50, 'Quarter 2'), (0.75, 'Quarter 3')]
    quarter_info = []

    for ratio, quarter in quarters:
        progress = 100/4 * (quarters.index((ratio, quarter))+1)
        pbar.progress(int(progress), f"Predicting {quarter}...")
        quarter_df = route_df[route_df["cum_distance"] <= route_df["cum_distance"].max() * ratio]
        route_agg_qtr, _ = process_gpx(quarter_df, season, time_of_day, watt_kilo, atl, ctl, pbar)
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
    center_lat, center_long = route_df["position_lat"].mean(), route_df["position_long"].mean()

    m = folium.Map(
        location=[center_lat, center_long],
        zoom_start=15,
        control_scale=True
    )

    route_df["slope_color"] = route_df["slope_color"].replace("downhill", "grey")

    for idx in range(1, len(route_df)):
        folium.PolyLine(
            [(route_df["position_lat"].iloc[idx - 1], route_df["position_long"].iloc[idx - 1]),
             (route_df["position_lat"].iloc[idx], route_df["position_long"].iloc[idx])],
            color=route_df["slope_color"].iloc[idx],
            weight=2.5,
            opacity=0.8
        ).add_to(m)

    # Add start marker
    folium.Marker(
        location=[route_df["position_lat"].iloc[0], route_df["position_long"].iloc[0]],
        popup="Start",
        icon=folium.Icon(color="green")
    ).add_to(m)

    # Add quarter markers with time and distance
    for idx, row in prediction.iterrows():
        icon_number = str(idx + 1)
        folium.Marker(
            location=[row["position_lat"], row["position_long"]],
            popup=f"{row['time_str']} - {row['distance']} km",
            icon=folium.Icon(icon=icon_number, prefix="fa", color="red")
        ).add_to(m)

    return st_folium(
        m,
        height=800,
        width=1500,
        returned_objects=[]
    )
