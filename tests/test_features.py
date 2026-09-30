"""
Unit tests for feature engineering, heuristic regexes, and feature transformations.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    JobFeatureExtractor,
    MONEY_PATTERNS,
    CRYPTO_PATTERNS,
    URGENCY_PATTERNS,
    PERSONAL_EMAIL_PATTERNS,
    CHAT_APP_PATTERNS,
    UPFRONT_FEE_PATTERNS,
)


def test_heuristic_regexes():
    # Money & Wire
    text_money = "Send funds via Western Union or wire transfer of $2500."
    assert len(MONEY_PATTERNS.findall(text_money)) >= 2

    # Crypto
    text_crypto = "Payments are processed in bitcoin and USDT directly to your wallet."
    assert len(CRYPTO_PATTERNS.findall(text_crypto)) >= 2

    # Urgency
    text_urgency = "Urgently hiring! No experience needed, make money fast!"
    assert len(URGENCY_PATTERNS.findall(text_urgency)) >= 2

    # Personal email
    text_email = "Send your resume directly to hr.recruiter99@gmail.com immediately."
    assert len(PERSONAL_EMAIL_PATTERNS.findall(text_email)) >= 1

    # Chat app
    text_chat = "Contact our manager on Telegram @jobs_fast or Whatsapp for interview."
    assert len(CHAT_APP_PATTERNS.findall(text_chat)) >= 2

    # Upfront fees
    text_fee = "Requires purchase of starter kit and $150 refundable deposit."
    assert len(UPFRONT_FEE_PATTERNS.findall(text_fee)) >= 2


def test_feature_extractor_output():
    data = [
        {
            "title": "Data Analyst",
            "company_profile": "Fortune 500 tech firm specializing in cloud data.",
            "description": "Analyze datasets and create BI dashboards for clients.",
            "requirements": "SQL, Python, Tableau proficiency required.",
            "benefits": "Competitive 401k, comprehensive dental and vision.",
            "has_company_logo": 1,
            "has_questions": 1,
            "telecommuting": 0,
            "has_salary": 1.0,
            "salary_spread": 20000.0,
            "salary_log_min": 11.2,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
        },
        {
            "title": "URGENT WORK FROM HOME!!!",
            "company_profile": "",
            "description": "Earn $2000 weekly! Contact us on telegram. Send fee via wire transfer.",
            "requirements": "",
            "benefits": "",
            "has_company_logo": 0,
            "has_questions": 0,
            "telecommuting": 1,
            "has_salary": 0.0,
            "salary_spread": 0.0,
            "salary_log_min": 0.0,
            "employment_type": "Unknown",
            "required_experience": "Unknown",
            "required_education": "Unknown",
        },
    ]
    df = pd.DataFrame(data)

    extractor = JobFeatureExtractor()
    extractor.fit(df)
    features_df = extractor.transform(df)

    # Check non-null & correct shape
    assert features_df.shape == (2, len(extractor.feature_names_))
    assert not features_df.isna().any().any()

    # Legit row checks
    row0 = features_df.iloc[0]
    assert row0["has_company_profile"] == 1.0
    assert row0["has_company_logo"] == 1.0
    assert row0["total_scam_flags"] == 0.0

    # Scam row checks
    row1 = features_df.iloc[1]
    assert row1["has_company_profile"] == 0.0
    assert row1["has_company_logo"] == 0.0
    assert row1["total_scam_flags"] >= 2.0
    assert row1["exclamation_ratio"] > 0
