import streamlit as st
import requests
import pandas as pd
import io
import os

st.title("ML Model Prediction Interface")

# App description
st.write(
    """
This Streamlit app demonstrates how to build an interactive interface that communicates with a machine learning prediction API. The primary goal is to provide an example of integrating a UI with an API for ML predictions—not to serve as a tool for quickly generating Kaggle competition submission files.
"""
)

st.write("Upload a CSV file to get predictions from the API.")


# Use Streamlit secrets if present, otherwise default to local URLs
SPACESHIP_API_URL = st.secrets.get(
    "SPACESHIP_API_URL", "http://127.0.0.1:8000/spaceship-titanic/predict"
)
DIGITS_API_URL = st.secrets.get(
    "DIGITS_API_URL", "http://127.0.0.1:8000/digit-recognizer/predict"
)

model_options = {
    "Spaceship Titanic": SPACESHIP_API_URL,
    "Digit Recognizer": DIGITS_API_URL,
}

# Track selected model in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(model_options.keys())[0]


def on_model_change():
    st.session_state.uploaded_file_key += 1


# Track file_uploader key in session state
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0

selected_model = st.selectbox(
    "Select Prediction Model",
    options=list(model_options.keys()),
    index=list(model_options.keys()).index(st.session_state.selected_model),
    on_change=on_model_change,
    key="selected_model",
)
API_URL = model_options[selected_model]

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key=f"file_uploader_{st.session_state.uploaded_file_key}",
)

if uploaded_file is not None:

    # Show uploaded file preview
    st.write("Preview of uploaded file:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Reset file pointer and send to API
    uploaded_file.seek(0)
    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
    with st.spinner("Getting predictions from API..."):
        response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        result = response.json()
        result_df = pd.DataFrame(result)

        st.success("Predictions received!")
        st.dataframe(result_df)

        # Optionally, allow download with correct columns and names
        if set(["PassengerId", "blend_pred"]).issubset(result_df.columns):
            download_df = result_df[["PassengerId", "blend_pred"]].rename(
                columns={"PassengerId": "PassengerId", "blend_pred": "Transported"}
            )
        else:
            download_df = result_df
        csv = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions as CSV", csv, "predictions.csv", "text/csv"
        )
    else:
        st.error(f"API Error: {response.status_code}\n{response.text}")
        st.warning(
            "The uploaded file likely did not match the expected format for the selected model. Please check your file and model selection."
        )
