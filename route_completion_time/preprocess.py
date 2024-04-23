import numpy as np
import pandas as pd
from tqdm import tqdm

processed_activities_file = "../data/train/processed_activities.csv"
wellness_file = "../data/train/wellness.csv"
df_activity = pd.read_csv(processed_activities_file)

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


def process_activity_by_id(data, id):
    """
    Process an activity by id
    :param data: DataFrame with the activities
    :param id: id of the activity to process
    :return: DataFrame with the processed activity

    :rtype: pd.DataFrame
    """
    df = data[data["id"] == id][COLS_TO_KEEP].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by=["timestamp"], inplace=True)
    # Calculate slope with safe division
    df["altitude"] = df["altitude"].fillna(0)
    df["slope"] = np.where(df["distance"].diff() != 0, df["altitude"].diff() / df["distance"].diff() * 100, 0)
    df["slope"] = df["slope"].fillna(0)
    df["altitude_diff"] = df["altitude"].diff()
    df["elapsed_time"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["season"] = df["timestamp"].dt.month.map(YEAR_SEASONS)
    df["time_of_day"] = pd.cut(df["timestamp"].dt.hour, bins=TIME_OF_DAY_CUTS, labels=TIME_OF_DAY_LABELS)
    df["slope_color"] = pd.cut(df["slope"], bins=SLOPE_CUTS, labels=SLOPE_LABELS)

    return df


def process_all_activities(data):
    """
    Process all activities. It applies the function process_activity_by_id to all the activities
    :param data: DataFrame with the activities
    :return: DataFrame with all the processed activities

    :rtype: pd.DataFrame
    """
    print("Processing all activities")
    id_list = data["id"].unique()
    processed_activities = [process_activity_by_id(data, id) for id in tqdm(id_list)]
    df_agg = pd.concat(processed_activities)

    return df_agg


def process_wellness(data):

    return data


def aggregate_waypoints(data):
    """
    Aggregates all the waypoints of an activity to a single row per activity
    :param data: DataFrame with the activities
    :return: DataFrame with the aggregated activities

    :rtype: pd.DataFrame
    """
    print("Aggregating waypoints")
    # Aggregates all the waypoints of an activity
    # - distance = max distance
    # - ascent_meters = sum of positive altitude_diff
    # - elapsed_time = max elapsed_time
    # - season = most common season
    # - time_of_day = most common time_of_day

    df_agg = data.groupby("id").agg(
        distance=("distance", "max"),
        ascent_meters=("altitude_diff", lambda x: x[x > 0].sum()),
        elapsed_time=("elapsed_time", "max"),
        season=("season", lambda x: x.value_counts().idxmax()),
        time_of_day=("time_of_day", lambda x: x.value_counts().idxmax()),
    ).reset_index()

    # Calculate the percentage of each slope color in the activity per total distance
    df_slope_color = data.groupby(["id", "slope_color"], observed=False).agg(
        distance=("distance", "sum")
    ).reset_index()
    df_slope_color = df_slope_color.pivot(index="id", columns="slope_color", values="distance").fillna(0)
    df_slope_color = df_slope_color.div(df_slope_color.sum(axis=1), axis=0)
    df_slope_color.columns = [f"{col}_pctg" for col in df_slope_color.columns]
    df_agg = pd.merge(df_agg, df_slope_color, on="id")

    return df_agg


df_activity = process_all_activities(df_activity)
df_activity = aggregate_waypoints(df_activity)
df_activity.to_csv("preprocessed.csv")
