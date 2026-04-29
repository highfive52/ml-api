from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO

router = APIRouter(prefix="/spaceship-titanic", tags=["Spaceship Titanic"])


def set_models(models):
    router.models = models


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        models = router.models

        xgb_proba = models["xgb_tuned"].predict_proba(df)[:, 1]
        lgbm_proba = models["lgbm_tuned"].predict_proba(df)[:, 1]
        hgb_proba = models["hgb_tuned"].predict_proba(df)[:, 1]

        blend_proba = (xgb_proba + lgbm_proba + hgb_proba) / 3
        blend_pred = (blend_proba >= 0.5).astype(bool)

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
