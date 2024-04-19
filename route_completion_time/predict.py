import gpxpy
import haversine as hs
from haversine import Unit
import pandas as pd

gpx_path = "../data/gpx/ruta_balcon_axarquia.gpx"

def ingest_gpx(gpx_path):

    gpx_file = open(gpx_path, 'r')
    gpx = gpxpy.parse(gpx_file)

    waypoints = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                waypoints.append([point.latitude, point.longitude, point.elevation, point.time])

    data = pd.DataFrame(waypoints, columns=["latitude", "longitude", "elevation", "time"])
    data.rename(columns={"latitude": "position_lat", "longitude": "position_long", "elevation": "altitude", "time": "timestamp"}, inplace=True)
    data.drop("timestamp", axis=1, inplace=True)

    return data

def categorize_grade(df):
    cuts = [-float("inf"), -10, -5, 5, 10, 20, float("inf")]
    labels = ["hard_dh", "dh", "flat", "up", "hard_up", "extreme_up"]
    df["grade_cat"] = pd.cut(df["grade"], bins=cuts, labels=labels)
    return df

def process_gpx(data):

    for i in range(1, len(data)):
        lat1, lon1, alt1 = data.loc[i - 1, ["position_lat", "position_long", "altitude"]]
        lat2, lon2, alt2 = data.loc[i, ["position_lat", "position_long", "altitude"]]

        distance = hs.haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
        elevation = alt2 - alt1
        grade = elevation / distance

        data.loc[i, "distance"] = distance
        data.loc[i, "grade"] = grade


    data.fillna(0, inplace=True)
    data = categorize_grade(data)

    # make something like this
    # grade_grouped = df_activity.groupby(["id", "grade_cat"], observed=False)["diff_meters"].sum()
    # grade_grouped = grade_grouped.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
    # grade_grouped.columns = [f"{col}_pctg" for col in grade_grouped.columns]
    # df_agg = df_agg.join(grade_grouped)


    # Now generate the final values
    # Create the columns
    # - distance_meters
    # - ascent_meters
    # - grade_cat

    distance_meters = data["distance"].sum()
    ascent_meters = data["altitude"].diff().apply(lambda x: x if x > 0 else 0).sum()
    # Per each

    return {
        "distance_meters": distance_meters,
        "ascent_meters": ascent_meters,

    }




SEASON = "summer"
TIME_OF_DAY = "morning"


df = ingest_gpx(gpx_path)
route_data = process_gpx(df)
print(route_data)
