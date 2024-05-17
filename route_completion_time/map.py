import folium
from streamlit_folium import st_folium
from folium.features import DivIcon


def create_map_from_prediction(prediction_df, route_df):
    best_model = prediction_df.sort_values("r2_score", ascending=False).iloc[0]["model"]
    prediction_df = prediction_df[prediction_df["model"] == best_model]

    prediction_df = prediction_df.sort_values("prediction_seconds")


    center_lat = route_df["latitude"].mean()
    center_long = route_df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_long], zoom_start=15, control_scale=True
    )

    # Change slopecolor "dh" and "dh_extreme" to blue
    route_df["slope_color"] = route_df["slope_color"].replace("dh", "blue")
    route_df["slope_color"] = route_df["slope_color"].replace("dh_extreme", "blue")

    for idx in range(1, len(route_df)):
        folium.PolyLine(
            [
                (
                    route_df["latitude"].iloc[idx - 1],
                    route_df["longitude"].iloc[idx - 1],
                ),
                (route_df["latitude"].iloc[idx], route_df["longitude"].iloc[idx]),
            ],
            color=route_df["slope_color"].iloc[idx],
            weight=2.5,
            opacity=0.8,
        ).add_to(m)

    i = 0
    for idx, row in prediction_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        time_seconds = row["prediction_seconds"]
        hours, remainder = divmod(time_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        time_str = f"{hours}h {minutes}m"
        popup_text = f"Prediction: {time_str}"


        icon_number = str(i + 1)
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(icon=icon_number, prefix="fa", color="red")
        ).add_to(m)
        i += 1

    # Add start. Blue marker
    folium.Marker(
        location=[route_df["latitude"].iloc[0], route_df["longitude"].iloc[0]],
        popup="Start",
        icon=folium.Icon(icon="play", color="blue")
    ).add_to(m)

    return m


def display_map(route_df, prediction_df):
    folium_map = create_map_from_prediction(route_df, prediction_df)
    st_folium(folium_map, height=800, width=1500, returned_objects=[])
