from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).with_name("model.joblib")

app = FastAPI(
    title="ML Malignancy Prediction API",
    version="1.0.0",
    description="API for breast cancer malignancy classification.",
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature values keyed by model feature name.")


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, float]] = Field(..., min_length=1, description="Records to score.")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str


def _read_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    bundle = joblib.load(model_path)
    if "pipeline" not in bundle:
        raise ValueError("Model bundle is missing the 'pipeline' key.")
    return bundle


@lru_cache(maxsize=1)
def get_model_bundle() -> dict[str, Any]:
    return _read_model_bundle()


def get_metadata() -> dict[str, Any]:
    try:
        return get_model_bundle().get("meta", {})
    except Exception:
        return {}


def expected_features() -> list[str]:
    return list(get_metadata().get("features", []))


def build_frame(records: list[dict[str, float]]) -> pd.DataFrame:
    features = expected_features()
    if not features:
        return pd.DataFrame(records)

    missing_by_row = {
        index: sorted(set(features) - set(record))
        for index, record in enumerate(records)
        if set(features) - set(record)
    }
    if missing_by_row:
        raise HTTPException(
            status_code=400,
            detail={"message": "Missing required model features.", "missing_by_row": missing_by_row},
        )

    return pd.DataFrame(records)[features]


def score_records(records: list[dict[str, float]]) -> list[PredictionResponse]:
    try:
        model = get_model_bundle()["pipeline"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    frame = build_frame(records)
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = model.predict(frame)
    target_names = get_metadata().get("target_names", ["malignant", "benign"])

    responses: list[PredictionResponse] = []
    for prediction, probability in zip(predictions, probabilities):
        label = target_names[int(prediction)] if int(prediction) < len(target_names) else str(int(prediction))
        responses.append(
            PredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                label=label,
            )
        )
    return responses


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    meta = get_metadata()
    if not meta:
        raise HTTPException(status_code=503, detail="Model metadata is not available.")
    return meta


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    return score_records([payload.features])[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest) -> list[PredictionResponse]:
    return score_records(payload.records)
