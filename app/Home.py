import datetime
import os
import sys
import tempfile
import threading
import time

import pandas as pd
import streamlit as st

import env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.route_time.predict_route_time import make_predictions
from app.route_time.train_route_time import main as train_main, MODEL_METRICS_SAVE_PATH
from app.fetch_data import main as fetch_data_main, fetch_wellness
from app.map import display_map, create_elevation_profile_plot
from app.intervals import Intervals



def load_model_metrics():
    try:
        model_metrics = pd.read_csv(MODEL_METRICS_SAVE_PATH)
    except FileNotFoundError:
        st.warning("No model metrics found. Please train a model first.")
        return None
    # Return which row has max r2_score
    return model_metrics[model_metrics["r2_score"] == model_metrics["r2_score"].max()]


def get_last_wellness():
    start_date = datetime.datetime.strptime(env.START_DATE, "%d/%m/%Y").date()
    today = datetime.date.today()
    icu = Intervals(env.ATHLETE_ID, env.API_KEY)
    try:
        wellness = fetch_wellness(icu, start_date, today)
        return wellness
    except FileNotFoundError:
        st.warning("No wellness data found. Please fetch wellness data first.")
        return None


# Get last row
wellness = get_last_wellness().iloc[-1]

ATL_INIT = int(wellness["atl"])
CTL_INIT = int(wellness["ctl"])
FTP_INIT = int(wellness["eftp"])
WEIGHT_KG_INIT = int(wellness["weight"])
MEAN_TEMP_C_INIT = 20
TIME_OF_DAY_INIT = 10
REVERSE_GPX = False

# Set page config to wide
st.set_page_config(page_title="MTB - IA", page_icon=":mountain_bicyclist:", layout="wide")

# Custom CSS to adjust the main content width
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 75%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def check_and_run_tasks():
    while True:
        now = datetime.datetime.now()
        if now.hour == 0 and now.minute == 0:
            fetch_data_main()
            train_main()
            time.sleep(60)
        time.sleep(60)


# Start the background thread
threading.Thread(target=check_and_run_tasks, daemon=True).start()

####################################################################################

st.title("MTB - IA")
st.write("Predict the completion time for your mountain bike rides")

model_metrics = load_model_metrics()

if model_metrics is not None:
    st.write("Best model metrics:")
    st.write("R2 Score:", model_metrics["r2_score"].values[0])
    st.write("Mean Absolute Error:", model_metrics["mean_absolute_error"].values[0])
    st.write("Mean Squared Error:", model_metrics["mean_squared_error"].values[0])
    st.write("Model:", model_metrics["model"].values[0])
    st.write("Trained on:", model_metrics["timestamp"].values[0])

st.sidebar.title("Input Parameters")

FTP = st.sidebar.number_input(
    "Functional Threshold Power (FTP)", min_value=0, max_value=1000, value=FTP_INIT
)

Weight = st.sidebar.number_input(
    "Weight (kg)", min_value=0, max_value=200, value=WEIGHT_KG_INIT
)

atl = st.sidebar.number_input(
    "Acute Training Load (ATL)", min_value=0, max_value=200, value=ATL_INIT
)
ctl = st.sidebar.number_input(
    "Chronic Training Load (CTL)", min_value=0, max_value=200, value=CTL_INIT
)

mean_temp = st.sidebar.number_input(
    "Mean Temperature (°C)", min_value=-20, max_value=50, value=MEAN_TEMP_C_INIT
)

# Toggle for reverse
REVERSE_GPX = st.sidebar.checkbox("Reverse GPX", REVERSE_GPX)

####################################################################################

# Button for fetching data

if st.button("Fetch Data"):
    pbar_fetch = st.progress(0)
    message = st.empty()
    fetch_data_main(st_pbar=pbar_fetch, st_message=message)

if st.button("Train Model"):
    pb = st.progress(0)
    message = st.empty()
    train_main(st_pbar=pb, st_message=message)

####################################################################################

uploaded_file = st.file_uploader("Choose a GPX file", type="gpx")

if st.button("Process"):
    if uploaded_file is None:
        st.warning("Please upload a GPX file.")
    else:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(uploaded_file.getvalue())
        temp_file_path = t_file.name

        prediction_df, activity_df = make_predictions(
            temp_file_path, TIME_OF_DAY_INIT, mean_temp, FTP, Weight, atl, ctl, REVERSE_GPX
        )

        # Summarize the prediction
        cols_to_show = [
            "model",
            "r2_score",
            "quart",
            "prediction_parsed",
            "avg_speed_kmh",
            "distance",
            "altitude_gain"
        ]
        prediction_df["distance"] = prediction_df["distance"] / 1000
        prediction_df["distance"] = prediction_df["distance"].round(2).astype(str) + " km"
        st.dataframe(prediction_df[prediction_df["r2_score"] == prediction_df["r2_score"].max()][cols_to_show])

        # Elev profile
        elev_profile = create_elevation_profile_plot(activity_df)
        st.plotly_chart(elev_profile)

        # Interactive map
        display_map(prediction_df, activity_df)
