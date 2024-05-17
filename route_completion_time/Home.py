import streamlit as st
import tempfile
from predict import make_predictions
from map import display_map

ATL_INIT = 50
CTL_INIT = 50
FTP_INIT = 220
WEIGHT_KG_INIT = 80
MEAN_TEMP_C_INIT = 15
TIME_OF_DAY_INIT = 10

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

st.title("MTB - IA")
st.write("Predict the completion time for your mountain bike rides")

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

uploaded_file = st.file_uploader("Choose a GPX file", type="gpx")

if st.button("Process"):
    if uploaded_file is None:
        st.warning("Please upload a GPX file.")
    else:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(uploaded_file.getvalue())
        temp_file_path = t_file.name

        prediction_df, activity_df = make_predictions(
            temp_file_path, TIME_OF_DAY_INIT, mean_temp, FTP, Weight, atl, ctl
        )

        st.dataframe(prediction_df.sort_values("r2_score", ascending=False))
        # display_map(prediction_df, activity_df)
