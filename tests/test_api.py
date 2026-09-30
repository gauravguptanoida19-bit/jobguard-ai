"""
Unit tests for FastAPI endpoints: /health, /model-info, /predict, and schema validation.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_model


@pytest.fixture(scope="module")
def client():
    load_model()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    # If model_metrics.json exists, status must be 200
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "headline_metrics" in data
        assert "pr_auc" in data["headline_metrics"]
    else:
        assert response.status_code == 503


def test_predict_endpoint_valid_job(client):
    legit_payload = {
        "title": "Senior Machine Learning Engineer",
        "company_profile": "Leading AI research firm delivering computer vision and NLP solutions.",
        "description": "Design and deploy scalable machine learning inference pipelines using PyTorch and FastAPI.",
        "requirements": "5+ years experience in Python, PyTorch, Docker, Kubernetes, and automated testing.",
        "benefits": "Competitive base salary, equity options, 401(k) matching, health and dental.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Master's Degree",
        "industry": "Information Technology",
        "function": "Engineering",
        "salary_range": "140000-180000",
    }
    response = client.post("/predict", json=legit_payload)
    if response.status_code == 200:
        data = response.json()
        assert "fraud_probability" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]
        assert isinstance(data["is_fraudulent"], bool)
        assert isinstance(data["reasons"], list)
        assert len(data["reasons"]) > 0
    else:
        assert response.status_code == 503


def test_predict_endpoint_scam_job(client):
    scam_payload = {
        "title": "URGENT DATA ENTRY WORK FROM HOME!!!",
        "company_profile": "",
        "description": "Earn $3000 weekly! Contact us via Telegram @fast_cash. Send $100 registration fee via wire transfer.",
        "requirements": "",
        "benefits": "",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": "Unknown",
        "required_experience": "Unknown",
        "required_education": "Unknown",
    }
    response = client.post("/predict", json=scam_payload)
    if response.status_code == 200:
        data = response.json()
        assert data["fraud_probability"] > 0.40
        assert any("wire transfer" in r.lower() or "fee" in r.lower() or "profile" in r.lower() for r in data["reasons"])
    else:
        assert response.status_code == 503


def test_predict_invalid_schema(client):
    # Missing required field 'title'
    bad_payload = {
        "company_profile": "Some description without title",
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
