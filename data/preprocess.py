import os
import pandas as pd
import tqdm
from datetime import datetime
import shutil

from dotenv import load_dotenv
load_dotenv(dotenv_path="../env.py")
import intervals

activity_folder = "../data/processed_activities"
processed_dest_file = "../data/train/processed_activities.csv"
wellness_dest_file = "../data/train/wellness.csv"
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

def save_df(df, destination_file):
    """
    Save a dataframe to a file
    :param df: Dataframe to save
    :param destination_file: Destination file path
    :return: None
    """
    folder_path = "/".join(destination_file.split("/")[:-1])
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(destination_file, index=False)
    print("Saved activity data to ", destination_file)

def intervals_get_wellness(athlete_id, api_key, save_path, start_date="2020-01-01", end_date="today"):
    """
    Get wellness data from intervals.icu and save it to a csv file
    :param athlete_id: Athlete ID from intervals.icu
    :param api_key: API key from intervals.icu
    :param save_path: Path to save the wellness data as a csv file
    :param start_date: Start date of the wellness data
    :param end_date: End date of the wellness data

    :return: Wellness data as a pandas dataframe
    :rtype: pd.DataFrame
    """
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

    wellness_df.to_csv(save_path, index=False)
    print("Saved wellness data to ", save_path)

    return wellness_df

def load_activity_files(activity_folder):
    """
    Load activity files from a folder
    :param activity_folder: Folder containing activity files

    :return: Dataframe containing all activity files
    :rtype: pd.DataFrame
    """
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
    """
    Preprocess activity data
    :param activity_df: Dataframe containing activity data

    :return: Preprocessed activity data
    :rtype: pd.DataFrame
    """
    activity_df = (
        activity_df[columns_to_keep]
        .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
        .sort_values(by="timestamp")
    )

    return activity_df

def main():
    activity_df = load_activity_files(activity_folder)
    activity_df = preprocess_activity(activity_df)
    save_df(activity_df, processed_dest_file)
    intervals_get_wellness(ATHLETE_ID, API_KEY, wellness_dest_file)


ATHLETE_ID = os.getenv("ATHLETE_ID")
API_KEY = os.getenv("API_KEY")

main()
