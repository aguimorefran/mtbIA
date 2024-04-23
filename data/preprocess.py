import os
import pandas as pd
import tqdm
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(dotenv_path="../env.py")
import intervals

activity_folder = "../data/processed_activities"
processed_dest_file = "../data/processed_activities.csv"
wellness_dest_file = "../data/wellness.csv"
columns_to_keep = [
    "id",
    "position_lat",
    "position_long",
    "distance",
    "speed",
    "timestamp",
    "cadence",
    "power",
    "temperature",
    "altitude",
    "grade"
]

def intervals_get_wellness(athlete_id, api_key, start_date="2020-01-01", end_date="today"):
    # Start date to datetime.date
    start_date = pd.to_datetime(start_date).date()
    # Datetime can be None, "today", or a date string
    if end_date != "today":
        end_date = pd.to_datetime(end_date).date()
    if end_date == "today":
        end_date = datetime.now().date()
    icu = intervals.Intervals(athlete_id, api_key)
    wellness = icu.wellness(start_date, end_date)

    wellness_df = pd.DataFrame(wellness)
    wellness_df['eftp'] = wellness_df['sportInfo'].apply(lambda x: x[0]['eftp'] if len(x) > 0 else None)

    wellness_df = wellness_df[["id", "eftp", "ctl", "atl"]]
    wellness_df["date"] = pd.to_datetime(wellness_df["id"])
    wellness_df = wellness_df.drop(columns=["id"])

    wellness_df.to_csv("../data/wellness.csv", index=False)
    print("Saved wellness data to ../data/wellness.csv")

    return wellness_df

def load_activity_files(activity_folder):
    print("Loading activity files from", activity_folder)
    activity_files = os.listdir(activity_folder)
    print("Found", len(activity_files), "activity files")

    df = pd.DataFrame()
    for activity_file in tqdm.tqdm(activity_files):
        activity = pd.read_csv(activity_folder + "/" + activity_file)
        activity["id"] = activity_file[:-4]
        df = pd.concat([df, activity])

    ids = df["id"].unique()
    print("Created ", len(ids), "unique ids")
    print("Columns:", df.columns)

    return df


def preprocess_activity(activity_df):
    activity_df = (
        activity_df[columns_to_keep]
        .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
        .sort_values(by="timestamp")
    )

    return activity_df


def save_activity(activity_df, destination_file):
    activity_df.to_csv(destination_file, index=False)
    print("Saved activity data to ", destination_file)


def main():
    activity_df = load_activity_files(activity_folder)
    activity_df = preprocess_activity(activity_df)
    save_activity(activity_df, processed_dest_file)
    intervals_get_wellness(ATHLETE_ID, API_KEY, wellness_dest_file)


ATHLETE_ID = os.getenv("ATHLETE_ID")
API_KEY = os.getenv("API_KEY")

main()
