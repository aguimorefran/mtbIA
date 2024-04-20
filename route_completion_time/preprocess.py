import numpy as np
import pandas as pd
import tqdm

processed_activities_file = "../data/ready_data.csv"
df_activity = pd.read_csv(processed_activities_file)

COLS_TO_KEEP = [
    "id",
    "timestamp",
    "position_lat",
    "position_long",
    "distance",
    "altitude"
]

GRADE_CUTS = [-np.inf, -1, 4, 8, 12, 20, np.inf]
GRADE_LABELS = [
    "downhill",
    "green",
    "yellow",
    "orange",
    "red",
    "black"
]

YEAR_SEASONS = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
    12: "winter"
}

TIME_OF_DAY_CUTS = [0, 6, 12, 18, 24]
TIME_OF_DAY_LABELS = [
    "night",
    "morning",
    "afternoon",
    "evening"
]

def preprocess(data):
    print("Preprocessing data...")
    df = data.copy()
    df = df[COLS_TO_KEEP]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["elapsed"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
    df["season"] = df["timestamp"].dt.month.map(YEAR_SEASONS)
    df["time_of_day"] = pd.cut(df["timestamp"].dt.hour, bins=TIME_OF_DAY_CUTS, labels=TIME_OF_DAY_LABELS)
    df["distance_diff"] = df["distance"].diff()
    df["altitude_diff"] = df["altitude"].diff()
    df["grade"] = df["altitude_diff"] / df["distance_diff"]
    df["grade"] = df["grade"].fillna(0)
    df["altitude_diff"] = df["altitude_diff"].fillna(0)
    df["distance_diff"] = df["distance_diff"].fillna(0)
    df["grade_category"] = pd.cut(df["grade"], bins=GRADE_CUTS, labels=GRADE_LABELS)
    return df

def aggregate_by_id(data, contains_time):
    # New columns
    # - distance_meters = sum(distance)
    # - ascent_meters = sum of positive altitude_diff
    # - elapsed_minutes = max(elapsed) / 60
    # - season = mode(season)
    # - time_of_day = mode(time_of_day)
    # - {grade_cat}_pctg = sum(distance_diff[grade_category == grade_cat])

    print("Aggregating data...")
    if contains_time:
        df_agg = data.groupby("id").agg(
            distance_meters=("distance", "sum"),
            ascent_meters=("altitude_diff", lambda x: sum(x[x > 0])),
            elapsed_minutes=("elapsed", lambda x: max(x) / 60),
            season=("season", lambda x: x.mode().values[0]),
            time_of_day=("time_of_day", lambda x: x.mode().values[0])
        ).reset_index()
    else:
        df_agg = data.groupby("id").agg(
            distance_meters=("distance", "sum"),
            ascent_meters=("altitude_diff", lambda x: sum(x[x > 0])),
            season=("season", lambda x: x.mode().values[0]),
            time_of_day=("time_of_day", lambda x: x.mode().values[0])
        ).reset_index()

    grade_cat_pivot = data.pivot_table(observed=False, index='id', columns='grade_category', values='distance', aggfunc='sum', fill_value=0)
    grade_cat_pctg = grade_cat_pivot.div(grade_cat_pivot.sum(axis=1), axis=0)
    grade_cat_pctg.columns = [f'{col}_pctg' for col in grade_cat_pctg.columns]
    df_agg = df_agg.merge(grade_cat_pctg, left_on='id', right_index=True)


    return df_agg

def preprocess_and_aggregate(data, contains_time):
    df = preprocess(data)
    df = aggregate_by_id(df, contains_time)

    # Save in preprocessed.csv
    df.to_csv("preprocessed.csv", index=False)
    print("Preprocessed data saved in preprocessed.csv")

preprocess_and_aggregate(df_activity, contains_time=True)