import tempfile

import streamlit as st
from utils import load_model, preprocess_gpx, predict, plot_map

PAGE_TITLE = "Time Prediction"
atl = 30
ctl = 30
ftp_watts = 220
weight_kg = 90
time_of_day_options = ["Morning", "Afternoon", "Evening"]
time_of_day = ""
season_options = ["Spring", "Summer", "Fall", "Winter"]
season = ""

model = None

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="⏳",
)

st.sidebar.title("How it works")
st.sidebar.write("""
The application uses a ML model trained on my past trainings.
The model takes into account various factors such as the distance, elevation, and rider's fitness level to make its predictions.
When you input the GPX of the route, the model will use this information to make its predictions.
""")


def input_values():
    global atl, ctl, ftp_watts, weight_kg, time_of_day, season
    st.write("Please input the following values to get a more accurate prediction.")
    atl = st.number_input("ATL/Fatigue (Acute Training Load)", min_value=0, max_value=1000, value=atl)
    ctl = st.number_input("CTL/Fitness (Chronic Training Load)", min_value=0, max_value=1000, value=ctl)
    ftp_watts = st.number_input("FTP (Functional Threshold Power) in watts", min_value=0, max_value=1000,
                                value=ftp_watts)
    weight_kg = st.number_input("Weight in kilograms", min_value=0, max_value=200, value=weight_kg)
    time_of_day = st.selectbox("Time of day", time_of_day_options)
    season = st.selectbox("Season", season_options)


def load_model_stats():
    global model
    model, stats = load_model()

    st.write("Model statistics:")
    st.dataframe(stats.set_index('r2'))


st.title(PAGE_TITLE)

input_values()
load_model_stats()

uploaded_file = st.file_uploader("Choose a GPX file", type="gpx")

if st.button("Process"):

    if uploaded_file is None:
        st.error("Please upload a GPX file")
        st.stop()

    if model is None:
        st.error("Error loading model. Please try again.")
        st.stop()

    pbar = st.progress(0, "Processing GPX file...")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.getvalue())
    temp_file_path = tfile.name
    watts_per_kg = ftp_watts / weight_kg
    route_agg, route_df = preprocess_gpx(temp_file_path, season, time_of_day, watts_per_kg, atl, ctl)


    if route_agg is None:
        st.error("Error processing the GPX file. Please try again.")
        st.stop()

    pbar.progress(50, "Predicting time...")

    distance = route_agg["distance"].values[0].round(2)
    ascent_meters = route_agg["ascent_meters"].values[0].round(2)
    downhill_pctg = route_agg["downhill_pctg"].values[0]
    green_pctg = route_agg["green_pctg"].values[0]
    yellow_pctg = route_agg["yellow_pctg"].values[0]
    orange_pctg = route_agg["orange_pctg"].values[0]
    red_pctg = route_agg["red_pctg"].values[0]
    black_pctg = route_agg["black_pctg"].values[0]

    prediction = predict(
        model, route_agg, route_df, season, time_of_day, watts_per_kg, atl, ctl, pbar
    )
    # Round column distance to 2 decimal places and divide by 1000 to convert to kilometers
    prediction["distance"] = (prediction["distance"] / 1000).round(2)

    pbar.progress(100, "Done!")

    st.dataframe(prediction)

    result_map = plot_map(route_df, prediction)