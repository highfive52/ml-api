import joblib
import os
import torch

from ml.spaceship.spaceship_transformers import (
    LogicalImputeTransformer,
    FeatureEngineer,
    Log1pTransformer,
    to_str,
)

from ml.digits.mnist_model import MNISTModel


def load_spaceship_models(models_dir):
    tuned_models = {}

    model_names = ["xgb_tuned", "lgbm_tuned", "hgb_tuned"]

    for name in model_names:
        path = os.path.join(models_dir, "spaceship", f"{name}.pkl")
        tuned_models[name] = joblib.load(path)

    return tuned_models


def load_digit_model(models_dir):
    checkpoint = torch.load(
        os.path.join(models_dir, "digit_recognizer", "mnist_checkpoint.pth"),
        map_location="cpu",
    )

    model = MNISTModel(dropout_rate=checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
