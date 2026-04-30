import pytest
from fastapi.testclient import TestClient
from app import app
import io

client = TestClient(app)


@pytest.mark.parametrize(
    "endpoint,csv_content,filename",
    [
        (
            "/spaceship-titanic/predict",
            # Minimal valid spaceship CSV (adjust columns as needed for your model)
            "PassengerId,Cabin,Name,Age,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,CryoSleep,VIP,HomePlanet,Destination\n"
            "0001_01,B/0/S,John Doe,30,0,0,0,0,0,0,0,Earth,Mars\n",
            "test_spaceship.csv",
        ),
        (
            "/digit-recognizer/predict",
            # Minimal valid digit recognizer CSV (flattened 28x28 = 784 columns)
            ",".join([f"pixel{i+1}" for i in range(784)])
            + "\n"
            + ",".join(["0"] * 784)
            + "\n",
            "test_digits.csv",
        ),
    ],
)
def test_api_endpoints(endpoint, csv_content, filename):
    file = io.BytesIO(csv_content.encode("utf-8"))
    response = client.post(endpoint, files={"file": (filename, file, "text/csv")})
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
