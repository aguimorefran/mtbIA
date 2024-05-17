import folium
from streamlit_folium import st_folium


def create_map_from_prediction(prediction_df, route_df):
    center_lat = route_df["latitude"].mean()
    center_long = route_df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_long], zoom_start=15, control_scale=True
    )

    for idx in range(1, len(route_df)):
        folium.PolyLine(
            [
                (
                    route_df["latitude"].iloc[idx - 1],
                    route_df["longitude"].iloc[idx - 1],
                ),
                (route_df["latitude"].iloc[idx], route_df["longitude"].iloc[idx]),
            ],
            color="blue",
            weight=2.5,
            opacity=0.8,
        ).add_to(m)

    folium.Marker(
        location=[route_df["latitude"].iloc[0], route_df["longitude"].iloc[0]],
        popup="Start",
        icon=folium.Icon(color="green"),
    ).add_to(m)

    for idx, row in prediction_df.iterrows():
        print(row)
        icon_number = str(idx + 1)
        time_seconds = row["time_seconds"]
        hours, remainder = divmod(time_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        time_str = f"{hours}h {minutes}m"

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['quarter']}: {time_str}",
            icon=folium.Icon(icon=icon_number, prefix="fa", color="red"),
        ).add_to(m)

    return m


def display_map(route_df, prediction_df):
    folium_map = create_map_from_prediction(route_df, prediction_df)
    st_folium(folium_map, height=800, width=1500, returned_objects=[])
