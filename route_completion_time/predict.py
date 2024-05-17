import argparse
import datetime
import os
from io import BytesIO
import logging
import pandas as pd
import numpy as np
import fitparse
import pickle
from sklearn.preprocessing import StandardScaler
import gpxpy
import haversine as hs
from fetch_data import WELLNESS_COLS, SLOPE_LABELS, SLOPE_CUTS
from train import MODEL_SAVE_PATH, SCALER_SAVE_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def read_gpx(gpx_path):
    gpx_file = open(gpx_path, "r")
    gpx = gpxpy.parse(gpx_file)
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append(
                    [point.latitude, point.longitude, point.elevation, point.time]
                )
    df = pd.DataFrame(data, columns=["latitude", "longitude", "elevation", "time"])
    return df


def load_model_scaler(model_path, scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def calculate_distance_slope(activity_df):
    activity_df = activity_df.copy()

    activity_df["distance"] = 0.0

    for i in range(1, len(activity_df)):
        distance = hs.haversine(
            (activity_df.loc[i - 1, "latitude"], activity_df.loc[i - 1, "longitude"]),
            (activity_df.loc[i, "latitude"], activity_df.loc[i, "longitude"]),
            unit=hs.Unit.METERS,
        )

        activity_df.loc[i, "distance"] = distance

    activity_df["distance"] = activity_df["distance"].fillna(0)

    activity_df["altitude_diff"] = activity_df["altitude"].diff().fillna(0)

    diffs = activity_df["distance"].replace(0, np.nan)
    slopes = np.where(diffs != 0, activity_df["altitude_diff"] / diffs * 100, 0)

    max_slope = 100
    slopes = np.clip(slopes, -max_slope, max_slope)

    activity_df["slope"] = slopes
    activity_df["slope"] = activity_df["slope"].fillna(0)
    activity_df["distance_diff"] = activity_df["distance"].diff().fillna(0)

    return activity_df


def aggregate_activity(activity_df):
    logger.info("Aggregating activity data")
    activity_df = activity_df.rename(
        columns={
            "time": "timestamp",
            "elevation": "altitude",
        }
    )

    activity_df = calculate_distance_slope(activity_df)

    activity_df["slope_color"] = pd.cut(
        activity_df["slope"], bins=SLOPE_CUTS, labels=SLOPE_LABELS
    )

    df_slope_color = (
        activity_df.groupby(["slope_color"], observed=False)
        .agg(distance=("distance", "sum"))
        .reset_index()
    )
    df_slope_color["distance"] = (
        df_slope_color["distance"] / df_slope_color["distance"].sum()
    )

    activity_summary = (
        activity_df.agg(
            {
                "distance": "sum",
                "altitude_diff": lambda x: x[x > 0].sum(),
            }
        )
        .to_frame()
        .T.rename(
            columns={"distance": "distance_meters", "altitude_diff": "elevation_gain"}
        )
    )
    df_slope_color = df_slope_color.set_index("slope_color").T
    df_slope_color.columns = [f"{col}_pct" for col in df_slope_color.columns]
    df_slope_color = df_slope_color.reset_index(drop=True)

    combined_df = pd.concat([activity_summary, df_slope_color], axis=1)

    return combined_df


def main(gpx_path, hour_of_day, avg_temperature, watts, kilos, atl, ctl):

    logger.info("Reading GPX file")
    gpx_data = read_gpx(gpx_path)

    logger.info("Loading model")
    model, scaler = load_model_scaler(MODEL_SAVE_PATH, SCALER_SAVE_PATH)

    logger.info("Processing wellness data")
    wellness_data = pd.DataFrame(
        {
            "ctl_start": [ctl],
            "atl_start": [atl],
            "watt_kg": [watts / kilos],
            "avg_temperature": [avg_temperature],
            "hour_of_day": [hour_of_day],
        }
    )

    df_aggregated = aggregate_activity(gpx_data)

    x_pred = pd.concat([df_aggregated, wellness_data], axis=1)
    order = [
        "distance_meters",
        "elevation_gain",
        "hour_of_day",
        "atl_start",
        "ctl_start",
        "watt_kg",
        "avg_temperature",
        "dh_extreme_pct",
        "dh_pct",
        "green_pct",
        "yellow_pct",
        "orange_pct",
        "red_pct",
        "black_pct",
    ]

    x_pred = x_pred[order]

    logger.info("Scaling data")
    x_pred_scaled = scaler.transform(x_pred)

    logger.info("Predicting completion time")
    completion_time = model.predict(x_pred_scaled)
    logger.info("Predicted completion time in seconds: %.2f", completion_time[0])
    logger.info(
        "Predict time formatted: %s",
        str(datetime.timedelta(seconds=completion_time[0])),
    )


if __name__ == "__main__":
    GPX_PATH = "data/gpx/test.gpx"
    TIME_OF_DAY = 12
    AVG_TEMP = 25
    WATTS = 220
    KILOS = 90
    ATL = 50
    CTL = 50

    main(GPX_PATH, TIME_OF_DAY, AVG_TEMP, WATTS, KILOS, ATL, CTL)
