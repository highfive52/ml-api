from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import numpy as np
import os
import pandas as pd
from transformers import (
    LogicalImputeTransformer,
    FeatureEngineer,
    Log1pTransformer,
    to_str,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = FastAPI()

# Load models at startup
tuned_models = {}

model_names = ["xgb_tuned", "lgbm_tuned", "hgb_tuned"]

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

for name in model_names:
    path = os.path.join(models_dir, f"{name}.pkl")
    tuned_models[name] = joblib.load(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV file into DataFrame
        contents = await file.read()
        from io import StringIO

        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Predict
        xgb_proba = tuned_models["xgb_tuned"].predict_proba(df)[:, 1]
        lgbm_proba = tuned_models["lgbm_tuned"].predict_proba(df)[:, 1]
        hgb_proba = tuned_models["hgb_tuned"].predict_proba(df)[:, 1]
        blend_proba = (xgb_proba + lgbm_proba + hgb_proba) / 3
        blend_pred = (blend_proba >= 0.5).astype(bool)

        # Include PassengerId if present
        passenger_ids = (
            df["PassengerId"].tolist()
            if "PassengerId" in df.columns
            else list(range(len(blend_pred)))
        )

        return {
            "PassengerId": passenger_ids,
            "blend_proba": blend_proba.tolist(),
            "blend_pred": blend_pred.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
