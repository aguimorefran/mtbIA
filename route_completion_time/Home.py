import streamlit as st
import tempfile


from predict import make_prediction
from map import display_map

ATL_INIT = 50
CTL_INIT = 50
FTP_INIT = 220
WEIGHT_KG_INIT = 80
MEAN_TEMP_C_INIT = 15
TIME_OF_DAY_INIT = 10


st.set_page_config(page_title="MTB - IA", page_icon=":mountain_bicyclist:")
st.title("MTB - IA")
st.write("Predict the completion time for your mountain bike rides")

st.sidebar.title("Input Parameters")

FTP = st.sidebar.slider(
    "Functional Threshold Power (FTP)", min_value=0, max_value=1000, value=FTP_INIT
)

Weight = st.sidebar.slider(
    "Weight (kg)", min_value=0, max_value=200, value=WEIGHT_KG_INIT
)

atl = st.sidebar.slider(
    "Acute Training Load (ATL)", min_value=0, max_value=200, value=ATL_INIT
)

ctl = st.sidebar.slider(
    "Chronic Training Load (CTL)", min_value=0, max_value=200, value=CTL_INIT
)

mean_temp = st.sidebar.slider(
    "Mean Temperature (°C)", min_value=-20, max_value=50, value=MEAN_TEMP_C_INIT
)

uploaded_file = st.file_uploader("Choose a GPX file", type="gpx")

if st.button("Process"):
    if uploaded_file is None:
        st.warning("Please upload a GPX file.")
    else:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(uploaded_file.getvalue())
        temp_file_path = t_file.name

        prediction, activity_df = make_prediction(
            temp_file_path, TIME_OF_DAY_INIT, mean_temp, FTP, Weight, atl, ctl
        )

        prediction["distance_km"] = (prediction["distance_meters"] / 1000).round(2)
        prediction["avg_speed_kmh"] = prediction["avg_speed_kmh"].round(2)

        st.dataframe(prediction)
        display_map(prediction, activity_df)

        # Refresh the values
        FTP = st.sidebar.slider(
            "Functional Threshold Power (FTP)",
            min_value=0,
            max_value=1000,
            value=FTP_INIT,
        )

        Weight = st.sidebar.slider(
            "Weight (kg)", min_value=0, max_value=200, value=WEIGHT_KG_INIT
        )

        atl = st.sidebar.slider(
            "Acute Training Load (ATL)", min_value=0, max_value=200, value=ATL_INIT
        )

        ctl = st.sidebar.slider(
            "Chronic Training Load (CTL)", min_value=0, max_value=200, value=CTL_INIT
        )

        mean_temp = st.sidebar.slider(
            "Mean Temperature (°C)", min_value=-20, max_value=50, value=MEAN_TEMP_C_INIT
        )
