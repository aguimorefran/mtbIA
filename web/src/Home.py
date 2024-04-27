import streamlit as st
from streamlit.logger import get_logger

def run():
    st.set_page_config(
        page_title="mtb - IA",
        page_icon=":mountain_bicyclist:",
    )

    st.title("mtb - IA")
    st.write("Predict various metrics about your mountain bike rides")
    st.write("Please refer to the sidebar for more information")

    st.sidebar.title("About")
    st.sidebar.write("This is a simple web application that uses machine learning to predict various metrics about your mountain bike rides.")

if __name__ == "__main__":
    run()