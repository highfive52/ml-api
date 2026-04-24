# ML API & Streamlit Demo

This project demonstrates a robust workflow for serving machine learning predictions via a FastAPI backend and a Streamlit web interface. It is designed as a portfolio example for best practices in ML API deployment, modular code, and interactive UI integration.

## Features

- **FastAPI Backend**: Serves ML model predictions via a RESTful API.
- **Streamlit Frontend**: Upload CSV files, select models, and view/download predictions interactively.
- **Custom Transformers**: Modularized for reliable model serialization and deployment.
- **Pre-commit Hooks**: Enforces code formatting and quality checks.

## Project Structure

```
api-ml/
├── api/
│   ├── src/
│   │   ├── main.py           # FastAPI app
│   │   ├── transformers.py   # Custom transformers
├── streamlit/
│   └── src/
│       └── app.py            # Streamlit UI
├── requirements.txt          # Pinned dependencies
├── .pre-commit-config.yaml   # Pre-commit hooks
└── README.md                 # Project overview
```

## Quickstart

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd api-ml
   ```

2. **Set up the environment**
   ```sh
   python -m venv .venv
   .venv/Scripts/activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Install pre-commit hooks**
   ```sh
   pip install pre-commit
   pre-commit install
   ```

4. **Start the API**
   ```sh
   uvicorn api.src.main:app --reload
   # or from api/src
   # uvicorn main:app --reload
   ```

5. **Start the Streamlit UI**
   ```sh
   streamlit run streamlit/src/app.py
   ```

6. **Test the API**
   ```sh
   python api/tests/test_api.py
   ```

## How it Works

- Upload a CSV file in the Streamlit UI.
- The file is sent directly to the FastAPI backend for prediction.
- The API returns predictions, which are displayed and downloadable in the UI.
- The system is modular—add more models and endpoints as needed.

## Why CSV Uploads?

- CSV is the standard for tabular ML data and batch predictions.
- The API handles all parsing and validation, ensuring consistency and reliability.
- No need to convert to JSON—keeps the workflow simple and robust.

## Not for Kaggle Submissions

This project is for demonstration and portfolio purposes. It is not intended as a tool for quickly generating Kaggle competition submission files.