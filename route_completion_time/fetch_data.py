import argparse
import datetime
import os
from io import BytesIO

import env
import fitparse
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

from intervals import Intervals

WELLNESS_COLS = [
    "ctl_start",
    "atl_start",
    "atl",
    "ctl",
    "date",
    "weight",
    "watt_kg",
]

SLOPE_CUTS = [-np.inf, -15, -1, 4, 8, 12, 20, np.inf]
SLOPE_LABELS = ["dh_extreme", "dh", "green", "yellow", "orange", "red", "black"]
SAVE_PATH = "data/activity_data.csv"

TODAY_DATE = datetime.date.today()


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


def fetch_and_combine_activity_data(icu, activity_ids):
    logger.info("Fetching activity data for %s activities", len(activity_ids))
    dfs = []

    for idx, activity_id in enumerate(activity_ids):
        try:
            result = retrieve_activity_data(icu, activity_id)
        except Exception as e:
            logger.error("Error fetching activity data for %s: %s", activity_id, e)
            continue
        if not result.empty:
            result["activity_id"] = activity_id
            result["hour_of_day"] = result["timestamp"].dt.hour
            dfs.append(result)

        if idx % (len(activity_ids) // 4) == 0:
            logger.info("Progress: %s/%s", 25 * (idx // (len(activity_ids) // 4)), 100)

    return pd.concat(dfs, ignore_index=True)


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
        activity_df["timestamp"] = pd.to_datetime(activity_df["timestamp"])
        activity_df.sort_values(by=["activity_id", "timestamp"], inplace=True)

        activity_df["altitude_diff"] = activity_df["altitude"].diff().fillna(0)
        diffs = activity_df["distance"].diff()
        slopes = np.where(diffs != 0, activity_df["altitude_diff"] / diffs * 100, 0)
        activity_df["slope"] = slopes
        activity_df["slope"].fillna(0)

        initial_time = activity_df["timestamp"].iloc[0]
        activity_df["elapsed_time"] = (
            activity_df["timestamp"] - initial_time
        ).dt.total_seconds()

        activity_df["slope_color"] = pd.cut(
            activity_df["slope"], bins=SLOPE_CUTS, labels=SLOPE_LABELS
        )

        activity_df["date"] = activity_df["timestamp"].dt.date.astype(str)
        wellness_df["date"] = wellness_df["date"].astype(str)
        agg_df = pd.merge(activity_df, wellness_df, on="date", how="left")

        agg_df = (
            agg_df.groupby(["activity_id", "date"])
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

    except Exception as e:
        logger.error("Error summarizing activity data: %s", e)
        return pd.DataFrame()

    return agg_df


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
    wellness_data = fetch_wellness(icu, start_date, datetime.date.today())
    activity_ids = [
        activity["id"] for activity in icu.activities(start_date, datetime.date.today())
    ]
    activity_data = fetch_and_combine_activity_data(icu, activity_ids)
    summarized_data = summarize_activity_data(activity_data, wellness_data)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    summarized_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
