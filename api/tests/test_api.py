import requests
import pandas as pd

ENDPOINTS = [
    # {
    #     "name": "Spaceship Titanic",
    #     "url": "http://127.0.0.1:8000/spaceship-titanic/predict",
    #     "csv_path": "data/test_spaceship.csv",
    #     "file_name": "test_spaceship.csv",
    # },
    {
        "name": "Spaceship Titanic",
        "url": "https://ml-api-ny6y.onrender.com/spaceship-titanic/predict",
        "csv_path": "data/test_spaceship.csv",
        "file_name": "test_spaceship.csv",
    },
    {
        "name": "Digit Recognizer",
        "url": "http://127.0.0.1:8000/digit-recognizer/predict",
        "csv_path": "data/test_digits.csv",
        "file_name": "test_digits.csv",
    },
]


def test_endpoint(endpoint):
    print(f"\nTesting {endpoint['name']} endpoint...")
    with open(endpoint["csv_path"], "rb") as f:
        files = {"file": (endpoint["file_name"], f, "text/csv")}
        response = requests.post(endpoint["url"], files=files)

    print("Status Code:", response.status_code)
    try:
        result = response.json()
        if isinstance(result, dict):
            # Print keys and preview values
            for k, v in result.items():
                print(f"{k}: {str(v)[:100]}{'...' if len(str(v)) > 100 else ''}")
        else:
            df = pd.DataFrame(result)
            print(df.head())
    except Exception as e:
        print("Error parsing response:", e)
        print(response.text)


if __name__ == "__main__":
    for endpoint in ENDPOINTS:
        test_endpoint(endpoint)
