"""
Smoke test verifying end-to-end model training, pipeline export, and inference on a sample.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.train import train_and_compare
from src.models.pipeline import JobGuardPipeline


def test_smoke_training_run(tmp_path):
    # Run rapid subsample training (250 samples) to isolated tmp_path
    smoke_model_path = tmp_path / "smoke_model.joblib"
    pipeline, summary = train_and_compare(
        sample_size=250,
        model_save_path=smoke_model_path,
        save_reports=False,
    )

    assert pipeline is not None
    assert "headline_metrics" in summary
    assert "pr_auc" in summary["headline_metrics"]

    # Verify model artifact saved on disk
    assert smoke_model_path.exists()

    # Verify pipeline loading and inference
    loaded = JobGuardPipeline.load(smoke_model_path)
    sample_posting = {
        "title": "Software Test Engineer",
        "company_profile": "Reputable tech organization with 500+ employees.",
        "description": "Write automated Python pytest and Playwright test suites.",
        "requirements": "Proficiency in pytest, git, CI/CD pipelines.",
        "benefits": "Competitive salary, 401k match, health insurance.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
    }

    probs = loaded.predict_proba([sample_posting])
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)

    preds = loaded.predict([sample_posting])
    assert preds[0] in (0, 1)
