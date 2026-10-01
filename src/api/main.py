"""
FastAPI Serving Application for JobGuard AI.
Exposes endpoints for job fraud classification, plain-English reasoning,
model evaluation telemetry, and service health.
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    JobPostingInput,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    UserProfile,
)
from src.models.pipeline import JobGuardPipeline
from src.explain.reasons import generate_plain_english_reasons

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
METRICS_PATH = Path("reports/model_metrics.json")
APP_VERSION = "0.1.0"
AUTH_SECRET = os.getenv("AUTH_SECRET", "jobguard-secure-auth-secret-key-2026")

DEMO_ACCOUNTS = {
    "analyst@jobguard.ai": {
        "id": "usr_analyst_01",
        "name": "Alex Morgan",
        "role": "Lead Fraud Analyst",
        "org": "JobGuard Threat Intelligence",
    },
    "auditor@jobguard.ai": {
        "id": "usr_auditor_02",
        "name": "Jordan Lee",
        "role": "Senior Compliance Auditor",
        "org": "JobGuard Trust & Safety",
    },
}


def create_access_token(email: str, role: str) -> str:
    """Generate a signed, URL-safe authentication token."""
    timestamp = str(int(time.time()))
    payload = f"{email}|{role}|{timestamp}"
    sig = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:24]
    token_str = f"{payload}|{sig}"
    return base64.urlsafe_b64encode(token_str.encode()).decode()


def decode_access_token(token: str) -> Optional[dict]:
    """Validate and unpack a signed authentication token."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split("|")
        if len(parts) != 4:
            return None
        email, role, timestamp, sig = parts
        payload = f"{email}|{role}|{timestamp}"
        expected_sig = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:24]
        if not hmac.compare_digest(sig, expected_sig):
            return None
        return {"email": email, "role": role, "timestamp": timestamp}
    except Exception:
        return None


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

# Enable CORS for Next.js frontend (local and deployed cloud domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://.*(onrender\.com|vercel\.app|localhost|127\.0\.0\.1).*",
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


@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
def login(creds: LoginRequest):
    """
    Authenticate an analyst or compliance auditor.
    Supports official demo accounts or dynamic registration for custom credentials.
    """
    email_clean = creds.email.strip().lower()

    if email_clean in DEMO_ACCOUNTS:
        account = DEMO_ACCOUNTS[email_clean]
        user = UserProfile(
            id=account["id"],
            email=email_clean,
            name=account["name"],
            role=account["role"],
            organization=account["org"],
        )
    else:
        # Dynamic profile generation for custom analyst logins
        local_part = email_clean.split("@")[0]
        name_derived = " ".join(part.capitalize() for part in local_part.replace(".", " ").replace("_", " ").split())
        user = UserProfile(
            id=f"usr_{abs(hash(email_clean)) % 100000:05d}",
            email=email_clean,
            name=name_derived or "Security Analyst",
            role="Fraud Investigator",
            organization="JobGuard Security Operations",
        )

    token = create_access_token(user.email, user.role)
    return LoginResponse(access_token=token, token_type="bearer", user=user)


@app.get("/auth/me", response_model=UserProfile, tags=["Authentication"])
def get_current_user(authorization: Optional[str] = Header(default=None)):
    """
    Retrieve profile of the currently authenticated analyst using Bearer token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected 'Bearer <token>'.",
        )

    token = authorization.split("Bearer ", 1)[1].strip()
    data = decode_access_token(token)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, tampered, or expired access token.",
        )

    email = data["email"]
    if email in DEMO_ACCOUNTS:
        account = DEMO_ACCOUNTS[email]
        return UserProfile(
            id=account["id"],
            email=email,
            name=account["name"],
            role=account["role"],
            organization=account["org"],
        )

    local_part = email.split("@")[0]
    name_derived = " ".join(part.capitalize() for part in local_part.replace(".", " ").replace("_", " ").split())
    return UserProfile(
        id=f"usr_{abs(hash(email)) % 100000:05d}",
        email=email,
        name=name_derived or "Security Analyst",
        role=data.get("role", "Fraud Investigator"),
        organization="JobGuard Security Operations",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
