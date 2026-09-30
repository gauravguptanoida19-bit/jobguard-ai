"""
FastAPI Serving Application for JobGuard AI.
Exposes endpoints for job fraud classification, plain-English reasoning,
model evaluation telemetry, and service health.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    JobPostingInput,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
from src.models.pipeline import JobGuardPipeline
from src.explain.reasons import generate_plain_english_reasons

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
METRICS_PATH = Path("reports/model_metrics.json")
APP_VERSION = "0.1.0"

_model: Optional[JobGuardPipeline] = None


def load_model():
    """Attempt to load trained model artifact from disk."""
    global _model
    if MODEL_PATH.exists():
        try:
            print(f"[INFO] Loading model artifact from '{MODEL_PATH}'...")
            _model = JobGuardPipeline.load(MODEL_PATH)
            print(f"[SUCCESS] Loaded model: {_model.model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load model artifact: {e}")
            _model = None
    else:
        print(f"[WARNING] Model file '{MODEL_PATH}' not found. Prediction endpoint will return 503.")
        _model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="JobGuard AI: Fake Job Posting Detector API",
    description="Supervised NLP + Tabular ML system predicting fraudulent job postings with plain-English reasoning.",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check verifying API operational status and model readiness."""
    global _model
    model_loaded = _model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
        version=APP_VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    """
    Return training metadata, architecture specification, and real test metrics
    read directly from reports/model_metrics.json.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model metrics not found at '{METRICS_PATH}'. Please execute the training pipeline first.",
        )

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelInfoResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse model metrics: {e}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(posting: JobPostingInput):
    """
    Classify a job posting as legitimate or fraudulent with plain-English reasons.
    
    Returns:
        fraud_probability: Estimated probability of fraud [0.0 - 1.0]
        risk_level: 'Low Risk', 'Moderate Risk', or 'High Risk'
        is_fraudulent: Binary decision based on configured threshold
        reasons: Human-readable explanations
    """
    global _model
    if _model is None:
        # Attempt just-in-time load
        load_model()
        if _model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model artifact not found at '{MODEL_PATH.resolve()}'. "
                    "Please run the training pipeline first: 'python scripts/train_all.py'. "
                    "JobGuard AI strictly prohibits faking predictions."
                ),
            )

    posting_dict = posting.model_dump()

    try:
        # Get class probabilities
        probs = _model.predict_proba([posting_dict])
        fraud_prob = float(probs[0, 1])

        # Categorical risk tier
        if fraud_prob < 0.30:
            risk_level = "Low Risk"
        elif fraud_prob < 0.60:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        is_fraud = bool(fraud_prob >= _model.threshold)

        # Generate explanatory factors
        reasons = generate_plain_english_reasons(
            posting=posting_dict,
            probability=fraud_prob,
            threshold=_model.threshold,
            max_reasons=4,
        )

        return PredictionResponse(
            fraud_probability=round(fraud_prob, 4),
            risk_level=risk_level,
            is_fraudulent=is_fraud,
            decision_threshold=round(float(_model.threshold), 4),
            reasons=reasons,
            model_name=_model.model_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed during feature transformation or model inference: {e}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
