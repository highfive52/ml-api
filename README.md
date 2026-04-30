# API Endpoints

## Prediction Endpoints

- [Spaceship Titanic Prediction (POST /spaceship-titanic/predict)](https://HighFive52-ml-api.hf.space/spaceship-titanic/predict)
	- Upload a CSV file with passenger data to get survival predictions.

- [Digit Recognizer Prediction (POST /digit-recognizer/predict)](https://HighFive52-ml-api.hf.space/digit-recognizer/predict)
	- Upload a CSV file with flattened 28x28 pixel values to get digit predictions.

**Note:** The prediction endpoints expect a file upload (form field name: `file`). Use the Swagger UI or an HTTP client (like curl or Postman) to submit data.

# Included Model Files

This repository includes pre-trained model files in the `models/` directory:

- `models/digit_recognizer/mnist_checkpoint.pth`
- `models/spaceship/hgb_tuned.pkl`, `lgbm_tuned.pkl`, `xgb_tuned.pkl`

These models are provided for public use with the API endpoints. You may use them for inference, testing, or further development. Please ensure you comply with any relevant licenses or terms of use for the models.

# API Endpoints

When deployed on Hugging Face Spaces, access the API endpoints directly:

- [Health check](https://HighFive52-ml-api.hf.space/health)
- [Swagger UI (interactive docs)](https://HighFive52-ml-api.hf.space/docs)
- [Redoc documentation](https://HighFive52-ml-api.hf.space/redoc)

**Note:** The endpoints above are not linked from the Space’s main page. Use these URLs to interact with the API directly.

