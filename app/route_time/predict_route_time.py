import logging
import os
import pickle

import gpxpy
import haversine as hs
import numpy as np
import pandas as pd
from app.fetch_data import SLOPE_LABELS, SLOPE_CUTS

from app.route_time.train_route_time import MODEL_METRICS_SAVE_PATH, MODELS_SAVE_PATH, SCALER_SAVE_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def read_gpx(gpx_path):
    try:
        with open(gpx_path, "rb") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        data.append(
                            [point.latitude, point.longitude, point.elevation, point.time]
                        )
        df = pd.DataFrame(data, columns=["latitude", "longitude", "altitude", "time"])
        return df
    except Exception as e:
        logger.error("Failed to read GPX file: %s", e)
        raise


def load_models_and_scaler(models_dir, scaler_path):
    models = {}
    for model_name in ["RandomForest", "Lasso", "Ridge"]:
        model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return models, scaler


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
    activity_df["slope_color"] = pd.cut(activity_df["slope"], bins=SLOPE_CUTS, labels=SLOPE_LABELS)
    activity_df["distance_accumulated"] = activity_df["distance"].cumsum()

    return activity_df


def aggregate_activity(activity_df):
    logger.info("Aggregating activity data")
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


def make_predictions(gpx_path, hour_of_day, avg_temperature, watts, kilos, atl, ctl, reverse=False):
    logger.info("Reading GPX file")
    gpx_data = read_gpx(gpx_path)

    if reverse:
        gpx_data = gpx_data.iloc[::-1].reset_index(drop=True)

    logger.info("Loading models and scaler")
    models, scaler = load_models_and_scaler(MODELS_SAVE_PATH, SCALER_SAVE_PATH)
    logger.info("Calculating distances and slopes")
    gpx_data = calculate_distance_slope(gpx_data)

    total_distance = gpx_data["distance"].sum()
    segments = [0.25, 0.50, 0.75, 1.0]
    segment_distances = [total_distance * segment for segment in segments]

    predictions = []

    for i, segment_distance in enumerate(segment_distances):
        segment_data = gpx_data[gpx_data["distance"].cumsum() <= segment_distance]
        df_aggregated = aggregate_activity(segment_data)
        logger.info("Aggregated activity data for segment %d", i + 1)
        wellness_data = pd.DataFrame(
            {
                "ctl_start": [ctl],
                "atl_start": [atl],
                "watt_kg": [watts / kilos],
                "avg_temperature": [avg_temperature],
                "hour_of_day": [hour_of_day],
            }
        )
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
        x_pred_scaled = scaler.transform(x_pred)

        for model_name, model in models.items():
            logger.info("Predicting completion time for segment %d with %s model", i + 1, model_name)
            completion_time = model.predict(x_pred_scaled)
            total_seconds = int(completion_time[0])
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
            avg_speed_ms = segment_distance / total_seconds
            avg_speed_kmh = avg_speed_ms * 3.6

            predictions.append({
                "quart": f"0-{int((i + 1) * 25)}%",
                "model": model_name,
                "prediction_seconds": total_seconds,
                "prediction_parsed": formatted_time,
                "avg_speed_kmh": avg_speed_kmh,
                "lat": segment_data["latitude"].iloc[-1],
                "lon": segment_data["longitude"].iloc[-1],
                "distance": segment_distance,
                "altitude_gain": segment_data["altitude_diff"].apply(lambda x: x if x > 0 else 0).sum(),

            })

            logger.info("%s model predicted completion time for segment %d in seconds: %d", model_name, i + 1,
                        total_seconds)
            logger.info("%s model predicted completion time for segment %d formatted: %s", model_name, i + 1,
                        formatted_time)

    metrics_df = pd.read_csv(MODEL_METRICS_SAVE_PATH)
    prediction_df = pd.DataFrame(predictions)
    result_df = pd.merge(metrics_df, prediction_df, on="model")

    # Keep model with latest timestamp
    latest_timestamp = result_df["timestamp"].max()
    result_df = result_df[result_df["timestamp"] == latest_timestamp]

    return result_df, gpx_data


if __name__ == "__main__":
    result_df, gpx_data = make_predictions(
        "data/gpx/test.gpx", 10, 15, 250, 80, 50, 50
    )
    print("Predictions and Metrics:")
    print(result_df)
    print("GPX Data with Slopes and Colors:")
    print(gpx_data)
