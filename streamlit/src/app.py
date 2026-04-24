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

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

model_options = {
    "Spaceship Titanic": os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
    # Add more models and their API URLs here as needed
}
selected_model = st.selectbox(
    "Select Prediction Model", options=list(model_options.keys())
)
API_URL = model_options[selected_model]

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
