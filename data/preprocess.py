import os
import pandas as pd
import tqdm

activity_folder = "data/processed_activities"
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


def save_activity(activity_df):
    activity_df.to_csv("data/processed_activities.csv", index=False)
    print("Saved activity data to data/processed_activities.csv")


def main():
    activity_df = load_activity_files(activity_folder)
    activity_df = preprocess_activity(activity_df)
    save_activity(activity_df)


main()
