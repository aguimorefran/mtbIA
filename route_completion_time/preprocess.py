import numpy as np
import pandas as pd

# Preprocessing
processed_activities_file = "../data/processed_activities.csv"
df_activity = pd.read_csv(processed_activities_file)

columns_to_keep = [
    "id",
    "timestamp",
    "position_lat",
    "position_long",
    "distance",
    "altitude",
    "grade"
]


def get_season(timestamp):
    month = timestamp.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"


def time_of_day(timestamp):
    hour = timestamp.hour
    if hour in range(6, 12):
        return "morning"
    elif hour in range(12, 18):
        return "afternoon"
    else:
        return "evening"


def ascent_meters(altitude):
    ascent = np.sum(np.maximum(altitude - np.roll(altitude, 1), 0))
    return ascent


def categorize_grade(df):
    cuts = [-float("inf"), -10, -5, 5, 10, 20, float("inf")]
    labels = ["hard_dh", "dh", "flat", "up", "hard_up", "extreme_up"]
    df["grade_cat"] = pd.cut(df["grade"], bins=cuts, labels=labels)
    return df


df_activity = df_activity[columns_to_keep]
df_activity["timestamp"] = pd.to_datetime(df_activity["timestamp"])
df_activity = df_activity.sort_values(by="timestamp")
df_activity = categorize_grade(df_activity)
df_activity["diff_meters"] = df_activity["distance"].diff()

# Group by id
df_agg = df_activity.groupby("id")
df_agg = df_agg.agg(
    duration_minutes=("timestamp", lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds() / 60),
    distance_meters=("distance", "max"),
    ascent_meters=("altitude", ascent_meters),
    time_of_day=("timestamp", lambda x: time_of_day(x.iloc[0])),
    year_season=("timestamp", lambda x: get_season(x.iloc[0]))
)

grade_grouped = df_activity.groupby(["id", "grade_cat"], observed=False)["diff_meters"].sum()
grade_grouped = grade_grouped.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
grade_grouped.columns = [f"{col}_pctg" for col in grade_grouped.columns]
df_agg = df_agg.join(grade_grouped)

df_agg.to_csv("preprocessed.csv")
