import pytest
from fastapi.testclient import TestClient

from app import main


class FakeProbabilities:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        rows, column = key
        if rows != slice(None, None, None) or column != 1:
            raise IndexError(key)
        return [row[1] for row in self.values]


class FakeModel:
    def predict_proba(self, frame):
        return FakeProbabilities([[0.8, 0.2] for _ in range(len(frame))])

    def predict(self, frame):
        return [0 for _ in range(len(frame))]


@pytest.fixture(autouse=True)
def fake_model_bundle(monkeypatch):
    main.get_model_bundle.cache_clear()
    monkeypatch.setattr(
        main,
        "get_model_bundle",
        lambda: {
            "pipeline": FakeModel(),
            "meta": {
                "features": ["mean radius", "mean texture"],
                "target_names": ["malignant", "benign"],
                "model_name": "fake",
            },
        },
    )
    yield


@pytest.fixture()
def client():
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata(client):
    response = client.get("/metadata")

    assert response.status_code == 200
    assert response.json()["features"] == ["mean radius", "mean texture"]


def test_predict(client):
    response = client.post(
        "/predict",
        json={"features": {"mean radius": 12.0, "mean texture": 15.0}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "probability": 0.2,
        "label": "malignant",
    }


def test_predict_rejects_missing_features(client):
    response = client.post(
        "/predict",
        json={"features": {"mean radius": 12.0}},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Missing required model features."


def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={
            "records": [
                {"mean radius": 12.0, "mean texture": 15.0},
                {"mean radius": 10.0, "mean texture": 13.0},
            ]
        },
    )

    assert response.status_code == 200
    assert len(response.json()) == 2
