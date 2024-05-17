import argparse
import datetime
import os
from io import BytesIO, StringIO

import env
import fitparse
import numpy as np
import pandas as pd
import logging
import sqlite3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

from intervals import Intervals

WELLNESS_COLS = [
    "ctl_start",
    "atl_start",
    "atl",
    "ctl",
    "date",
    "watt_kg",
]

SLOPE_CUTS = [-np.inf, -15, -1, 4, 8, 12, 20, np.inf]
SLOPE_LABELS = ["dh_extreme", "dh", "green", "yellow", "orange", "red", "black"]
SAVE_PATH = "data/activity_data.csv"
DB_PATH = "data/activities.db"

TODAY_DATE = datetime.date.today()


def initialize_db(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activities (
                activity_id TEXT PRIMARY KEY,
                data BLOB
            )
        """
        )
        conn.commit()


def load_activity_data(activity_id, db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM activities WHERE activity_id = ?", (activity_id,)
        )
        data = cursor.fetchone()
        return data[0] if data else None


def save_activity_data(activity_id, data, db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO activities (activity_id, data) VALUES (?, ?)",
                (activity_id, data),
            )
            conn.commit()
        except Exception as e:
            logger.error("Error saving activity data: %s", e)


def check_activity_data(activity_id, db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT activity_id FROM activities WHERE activity_id = ?", (activity_id,)
        )
        data = cursor.fetchone()
        return data[0] if data else None


def fetch_wellness(icu, start_date, end_date):
    logger.info("Fetching wellness data for %s to %s", start_date, end_date)

    try:
        wellnessData = icu.wellness(start_date, end_date)
        df = pd.DataFrame(wellnessData)
        df = process_wellness_data(df)
        return df
    except Exception as e:
        logger.error("Error fetching wellness data: %s", e)
        return pd.DataFrame()


def process_wellness_data(df):
    df["date"] = pd.to_datetime(df["id"])
    df = df.drop(columns=["id"])
    df = df.sort_values(by="date")
    df["eftp"] = df["sportInfo"].apply(lambda x: x[0]["eftp"] if len(x) > 0 else None)
    df = df.drop(columns=["sportInfo"])
    df["weight"] = df["weight"].ffill().bfill()
    df["eftp"] = df["eftp"].ffill().bfill()
    df["ctl_start"] = df["ctl"].shift(1)
    df["atl_start"] = df["atl"].shift(1)
    df["watt_kg"] = df["eftp"] / df["weight"]

    return df[WELLNESS_COLS]


def retrieve_activity_data(icu, activity_id):
    try:
        fit_bytes = icu.activity_fit_data(activity_id)
        if not fit_bytes:
            logger.error("No data received for activity %s", activity_id)
            return pd.DataFrame()

        fitfile = fitparse.FitFile(BytesIO(fit_bytes))

        records = []
        for record in fitfile.get_messages("record"):
            record_dict = {}
            for record_data in record:
                record_dict[record_data.name] = record_data.value
            records.append(record_dict)

        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logger.error("Error fetching activity data for %s: %s", activity_id, e)
        return pd.DataFrame()


def fetch_and_combine_activity_data(icu, activity_ids, db_path=DB_PATH):
    # TODO: CHECK IF ACTIVITY DAY BEFORE
    # TODO: CHECK WEATHER
    logger.info("Fetching activity data for %s activities", len(activity_ids))
    dfs = []

    for idx, activity_id in enumerate(activity_ids):
        data = load_activity_data(activity_id, db_path)
        if data:
            result = pd.read_json(StringIO(data))
            result["activity_id"] = activity_id
            result["hour_of_day"] = result["timestamp"].dt.hour
            dfs.append(result)
        else:
            try:
                result = retrieve_activity_data(icu, activity_id)
                if not result.empty:
                    save_activity_data(activity_id, result.to_json(), db_path)
                    result["activity_id"] = activity_id
                    result["hour_of_day"] = result["timestamp"].dt.hour
                    dfs.append(result)
            except Exception as e:
                logger.error("Error fetching activity data for %s: %s", activity_id, e)
                continue

        if idx % (len(activity_ids) // 4) == 0:
            logger.info("Progress: %s/%s", 25 * (idx // (len(activity_ids) // 4)), 100)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def aggregate_activity(activity_df, wellness_df):
    activity_df = activity_df.copy()
    activity_df["timestamp"] = pd.to_datetime(activity_df["timestamp"])
    activity_df.sort_values(by=["activity_id", "timestamp"], inplace=True)

    activity_df["distance"] = activity_df["distance"].fillna(0)
    activity_df["distance_diff"] = activity_df["distance"].diff().fillna(0)


    slopes = np.where(
        activity_df["distance_diff"] != 0,
        (activity_df["altitude_diff"] / activity_df["distance_diff"]) * 100,
        0,
    )

    max_slope = 100
    slopes = np.clip(slopes, -max_slope, max_slope)

    activity_df["slope"] = slopes
    activity_df["slope"] = activity_df["slope"].fillna(0)

    initial_time = activity_df["timestamp"].iloc[0]
    activity_df["elapsed_time"] = (
        activity_df["timestamp"] - initial_time
    ).dt.total_seconds()

    activity_df["slope_color"] = pd.cut(
        activity_df["slope"], bins=SLOPE_CUTS, labels=SLOPE_LABELS
    )

    activity_df["date"] = activity_df["timestamp"].dt.date.astype(str)
    wellness_df["date"] = wellness_df["date"].astype(str)
    merged_df = pd.merge(activity_df, wellness_df, on="date", how="left")

    agg_df = (
        merged_df.groupby(["activity_id", "date"])
        .agg(
            {
                "distance": "max",
                "altitude_diff": lambda x: x[x > 0].sum(),
                "elapsed_time": "max",
                "hour_of_day": "first",
                "atl_start": "first",
                "ctl_start": "first",
                "watt_kg": "first",
                "temperature": "mean",
            }
        )
        .reset_index()
        .rename(
            columns={
                "altitude_diff": "elevation_gain",
                "elapsed_time": "duration_seconds",
                "temperature": "avg_temperature",
                "distance": "distance_meters",
            }
        )
    )

    agg_df = agg_df.dropna()

    df_slope_color = (
        activity_df.groupby(["activity_id", "slope_color"], observed=False)
        .agg(distance=("distance", "sum"))
        .reset_index()
    )
    df_slope_color = df_slope_color.pivot(
        index="activity_id", columns="slope_color", values="distance"
    ).fillna(0)
    df_slope_color = df_slope_color.div(df_slope_color.sum(axis=1), axis=0)
    df_slope_color.columns = [f"{col}_pct" for col in df_slope_color.columns]
    agg_df = pd.merge(agg_df, df_slope_color, on="activity_id", how="left")

    return agg_df


def summarize_activity_data(activity_df, wellness_df):
    if activity_df.empty:
        return pd.DataFrame()

    logger.info(
        "Summarizing activity data for %s activities",
        activity_df["activity_id"].nunique(),
    )

    required_cols_activity = [
        "activity_id",
        "timestamp",
        "distance",
        "altitude",
    ]
    required_cols_wellness = ["date"]

    if not set(required_cols_activity).issubset(activity_df.columns):
        logger.error(
            "Error: Required columns missing in activity data: %s",
            required_cols_activity,
        )
        return pd.DataFrame()

    if not set(required_cols_wellness).issubset(wellness_df.columns):
        logger.error(
            "Error: Required columns missing in wellness data: %s",
            required_cols_wellness,
        )
        return pd.DataFrame()

    try:
        aggregated_results = []
        for activity_id in activity_df["activity_id"].unique():
            activity_data = activity_df.loc[
                activity_df["activity_id"] == activity_id
            ].copy()
            aggregated_activity = aggregate_activity(activity_data, wellness_df)
            aggregated_results.append(aggregated_activity)

        final_df = pd.concat(aggregated_results, ignore_index=True)

    except Exception as e:
        logger.error("Error summarizing activity data: %s", e)
        return pd.DataFrame()

    return final_df


def main(save_path=SAVE_PATH):
    logger.info("Starting data fetch")
    parser = argparse.ArgumentParser(description="Fetch wellness and activity data")
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for fetching data, format: dd/mm/yyyy",
    )

    parser.add_argument(
        "--athlete_id",
        type=int,
        help="Athlete ID for fetching data",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        help="API key for fetching data",
    )

    args = parser.parse_args()

    start_date = datetime.datetime.strptime(
        args.start_date or env.START_DATE, "%d/%m/%Y"
    ).date()
    icu = Intervals(env.ATHLETE_ID, env.API_KEY)
    initialize_db(DB_PATH)
    wellness_data = fetch_wellness(icu, start_date, datetime.date.today())
    activity_ids = [
        activity["id"] for activity in icu.activities(start_date, datetime.date.today())
    ]
    activity_data = fetch_and_combine_activity_data(icu, activity_ids, DB_PATH)
    summarized_data = summarize_activity_data(activity_data, wellness_data)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    summarized_data.to_csv(save_path, index=False)

    logger.info("Data fetch complete")


if __name__ == "__main__":
    main()
