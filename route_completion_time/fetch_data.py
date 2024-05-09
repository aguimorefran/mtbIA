import os
import sys
import argparse
import datetime
import pandas as pd
import tqdm
import fitparse
import numpy as np
from io import BytesIO
import concurrent.futures
from dotenv import load_dotenv
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
TIME_OF_DAY_CUTS = [0, 6, 12, 18, 24]
TIME_OF_DAY_LABELS = ["night", "morning", "afternoon", "evening"]

TODAY_DATE = datetime.date.today()
START_DATE = "02/12/2023"


def fetch_wellness(icu, start_date):
    global TODAY_DATE

    print("-" * 80)
    print(f"Fetching wellness data from {start_date} to {TODAY_DATE}")

    wellnessData = icu.wellness(start_date, TODAY_DATE)
    df = pd.DataFrame(wellnessData)
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
        print(f"Error fetching activity {activity_id}: {e}")
        return pd.DataFrame()


def fetch_and_combine_activity_data(icu, activity_ids):
    print("-" * 80)
    print(f"Fetching activity data for {len(activity_ids)} activities")
    dfs = []
    pbar = tqdm.tqdm(activity_ids)
    for activity_id in pbar:
        pbar.set_description(f"Fetching activity {activity_id}")
        result = retrieve_activity_data(icu, activity_id)
        if not result.empty:
            result["activity_id"] = activity_id
            result["hour_of_day"] = result["timestamp"].dt.hour
            dfs.append(result)

    return pd.concat(dfs, ignore_index=True)


def summarize_activity_data(activity_df, wellness_df):
    print("-" * 80)
    print("Summarizing activity data")
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

    activity_df["time_of_day"] = pd.cut(
        activity_df["timestamp"].dt.hour,
        bins=TIME_OF_DAY_CUTS,
        labels=TIME_OF_DAY_LABELS,
        right=False,
    )
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
                "time_of_day": "first",
                "hour_of_day": "first",
                "atl_start": "first",
                "ctl_start": "first",
                "weight": "first",
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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch wellness and activity data")
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for fetching data, format: dd/mm/yyyy",
    )
    parser.add_argument("--athlete_id", type=str, help="Athlete ID for the API")
    parser.add_argument("--api_key", type=str, help="API Key for accessing the API")
    args = parser.parse_args()

    start_date = args.start_date if args.start_date else START_DATE
    start_date = datetime.datetime.strptime(start_date, "%d/%m/%Y").date()

    ATHLETE_ID = args.athlete_id if args.athlete_id else os.getenv("ATHLETE_ID")
    API_KEY = args.api_key if args.api_key else os.getenv("API_KEY")

    if not ATHLETE_ID or not API_KEY:
        print(
            "Error: ATHLETE_ID and API_KEY must be provided either via command line or .env file."
        )
        sys.exit(1)

    icu = Intervals(ATHLETE_ID, API_KEY)

    wellness_data = fetch_wellness(icu, start_date)
    activity_list = icu.activities(start_date, TODAY_DATE)
    activity_id_list = [activity["id"] for activity in activity_list]
    fetched_activity_data = fetch_and_combine_activity_data(icu, activity_id_list)
    summarized_data = summarize_activity_data(fetched_activity_data, wellness_data)
    summarized_data.to_csv("activity_data.csv", index=False)


if __name__ == "__main__":
    main()
