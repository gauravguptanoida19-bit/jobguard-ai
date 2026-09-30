"""
Unit tests for data cleaning, text preprocessing, and salary parsing.
"""

import pandas as pd
import numpy as np
import pytest

from src.data.cleaner import (
    strip_html_and_clean_text,
    parse_salary_range,
    normalize_categoricals,
    clean_job_postings,
)


def test_strip_html_and_clean_text():
    html_input = "<p>Join our team at &amp; Company! <br>Apply at http://example.com/apply now.</p>"
    cleaned = strip_html_and_clean_text(html_input)
    assert "<p>" not in cleaned
    assert "<br>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "[URL]" in cleaned
    assert "Join our team at & Company!" in cleaned

    # Edge cases
    assert strip_html_and_clean_text(None) == ""
    assert strip_html_and_clean_text(np.nan) == ""
    assert strip_html_and_clean_text("") == ""
    assert strip_html_and_clean_text(12345) == "12345"


def test_parse_salary_range():
    # Valid range
    res = parse_salary_range("40000-60000")
    assert res["has_salary"] == 1.0
    assert res["salary_min"] == 40000.0
    assert res["salary_max"] == 60000.0
    assert res["salary_spread"] == 20000.0
    assert res["salary_log_min"] > 0

    # Reversed order
    res_rev = parse_salary_range("80000-50000")
    assert res_rev["salary_min"] == 50000.0
    assert res_rev["salary_max"] == 80000.0

    # Zero range
    res_zero = parse_salary_range("0-0")
    assert res_zero["has_salary"] == 0.0
    assert res_zero["salary_min"] == 0.0

    # Missing / Invalid
    res_none = parse_salary_range(None)
    assert res_none["has_salary"] == 0.0
    res_nan = parse_salary_range(np.nan)
    assert res_nan["has_salary"] == 0.0
    res_invalid = parse_salary_range("Negotiable / Competitive")
    assert res_invalid["has_salary"] == 0.0


def test_normalize_categoricals():
    df = pd.DataFrame({
        "employment_type": [None, "Full-time", np.nan],
        "required_experience": ["Entry level", "", "None"],
        "telecommuting": [1.0, np.nan, 0],
    })
    norm_df = normalize_categoricals(df)
    assert norm_df["employment_type"].tolist() == ["Unknown", "Full-time", "Unknown"]
    assert norm_df["required_experience"].tolist() == ["Entry level", "Unknown", "Unknown"]
    assert norm_df["telecommuting"].tolist() == [1, 0, 0]


def test_clean_job_postings_pipeline():
    sample_df = pd.DataFrame([
        {
            "job_id": 1,
            "title": "Software Engineer",
            "company_profile": "<p>A high-tech startup</p>",
            "description": "Develop scalable Python applications.",
            "requirements": "3+ years Python experience.",
            "benefits": "Health insurance &amp; 401k.",
            "salary_range": "100000-140000",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 0,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology",
            "function": "Engineering",
            "fraudulent": 0,
        },
        {
            "job_id": 2,
            "title": "Data Entry Clerk",
            "company_profile": None,
            "description": "Urgent work from home data entry. Earn $5000/week.",
            "requirements": "Must have computer.",
            "benefits": None,
            "salary_range": None,
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": None,
            "required_experience": None,
            "required_education": None,
            "industry": None,
            "function": None,
            "fraudulent": 1,
        },
    ])

    cleaned = clean_job_postings(sample_df, drop_duplicates=True, create_combined_text=True)

    assert "combined_text" in cleaned.columns
    assert "has_company_profile" in cleaned.columns
    assert "has_salary" in cleaned.columns
    assert "salary_min" in cleaned.columns

    # Verify first row
    assert cleaned.loc[0, "has_company_profile"] == 1
    assert cleaned.loc[0, "has_salary"] == 1.0
    assert cleaned.loc[0, "salary_min"] == 100000.0

    # Verify second row (fake)
    assert cleaned.loc[1, "has_company_profile"] == 0
    assert cleaned.loc[1, "has_salary"] == 0.0
    assert cleaned.loc[1, "employment_type"] == "Unknown"
    assert "Job Title: Data Entry Clerk" in cleaned.loc[1, "combined_text"]
