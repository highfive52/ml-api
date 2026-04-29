from fastapi import APIRouter, HTTPException, UploadFile, File
import io
import pandas as pd
import numpy as np
import torch
from typing import List

router = APIRouter(prefix="/digit-recognizer", tags=["Digit Recognizer"])


def set_digit_model(model):
    router.model = model


@router.post("/predict")
async def predict_digits_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Read CSV into DataFrame
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Each row is a flattened 28x28 image
        images = df.values.astype(np.float32).reshape(-1, 1, 28, 28)
        model = router.model

        BATCH_SIZE = 256  # Adjust as needed for your server's memory
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i : i + BATCH_SIZE]
                tensor = torch.tensor(batch)
                outputs = model(tensor)
                preds = outputs.argmax(dim=1).tolist()
                all_preds.extend(preds)

        # Add ImageID (row number starting from 1) and Label (predicted digit)
        result_df = pd.DataFrame(
            {"ImageID": list(range(1, len(all_preds) + 1)), "Label": all_preds}
        )
        return result_df.to_dict(orient="list")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
