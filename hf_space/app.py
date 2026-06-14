import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="WC 2026 Model Inference", version="1.0.0")

MODEL_DIR = Path(__file__).parent / "models"
API_TOKEN = os.environ.get("INFERENCE_API_TOKEN", "")
FEATURE_COLUMNS = [
    "chenh_lech_phong_do",
    "chenh_lech_ban_thang_tb",
    "chenh_lech_ban_thua_tb",
    "chenh_lech_doi_dau",
    "chenh_lech_fifa_rank",
    "chenh_lech_elo",
    "elo_expected_win_prob",
    "chenh_lech_kinh_nghiem_wc",
    "chenh_lech_thang_wc",
    "san_trung_lap",
    "is_host_team_A",
    "is_host_team_B",
]
MODEL_FILES = {
    "random_forest_baseline": "random_forest_baseline.pkl",
    "random_forest_tuned": "random_forest_tuned.pkl",
    "xgboost_baseline": "xgboost_baseline.pkl",
    "xgboost_tuned": "xgboost_tuned.pkl",
}
MODEL_CACHE = {}


class PredictRequest(BaseModel):
    model: str
    features: dict[str, float]


def authorize(authorization: str | None):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_model(model_name: str):
    if model_name not in MODEL_FILES:
        raise HTTPException(status_code=404, detail="Unknown model")
    if model_name not in MODEL_CACHE:
        model_path = MODEL_DIR / MODEL_FILES[model_name]
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="Model file is missing")
        MODEL_CACHE[model_name] = joblib.load(model_path)
    return MODEL_CACHE[model_name]


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODEL_CACHE)}


@app.get("/models")
def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {
        name: {"available": (MODEL_DIR / filename).exists(), "loaded": name in MODEL_CACHE}
        for name, filename in MODEL_FILES.items()
    }


@app.post("/predict")
def predict(payload: PredictRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    missing = [column for column in FEATURE_COLUMNS if column not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    model = load_model(payload.model)
    frame = pd.DataFrame([[payload.features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(frame)[0]
    classes = [int(value) for value in model.classes_]
    by_class = {label: float(probability) for label, probability in zip(classes, probabilities)}
    label = int(model.predict(frame)[0])

    return {
        "model": payload.model,
        "nhan_du_doan": label,
        "ti_le": {
            "p_a_thua": by_class.get(0, 0.0),
            "p_hoa": by_class.get(1, 0.0),
            "p_a_thang": by_class.get(2, 0.0),
        },
    }
