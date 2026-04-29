from fastapi import FastAPI
from pathlib import Path

from routers import spaceship, digits
from ml.common.model_loader import load_spaceship_models, load_digit_model

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
print(f"Base directory: {BASE_DIR}")
models_dir = BASE_DIR / "models"

# Load models
tuned_models = load_spaceship_models(models_dir)
digit_model = load_digit_model(models_dir)

# Inject into routers
spaceship.set_models(tuned_models)
digits.set_digit_model(digit_model)

# Register routers
app.include_router(spaceship.router)
app.include_router(digits.router)


@app.get("/health")
def health():
    return {"status": "ok"}
