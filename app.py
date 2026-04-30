from fastapi import FastAPI
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routers import spaceship, digits
from ml.common.model_loader import load_spaceship_models, load_digit_model

import joblib

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
print(f"Base directory: {BASE_DIR}")
models_dir = BASE_DIR / "models"
print(f"Models directory: {models_dir}")

joblib.load("models/spaceship/xgb_tuned.pkl")

# Load models
tuned_models = load_spaceship_models(models_dir)
digit_model = load_digit_model(models_dir)

# Inject into routers
spaceship.set_models(tuned_models)
digits.set_digit_model(digit_model)

# Register routers
app.include_router(spaceship.router)
app.include_router(digits.router)

# Mount static directory for favicon
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# Serve favicon.ico at root
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.ico")


@app.get("/health")
def health():
    return {"status": "ok"}
