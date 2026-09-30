"""
Feature Engineering module for JobGuard AI.
Extracts lexical, behavioral, regex heuristic, and structural features
from job posting text and metadata.
"""

import re
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Domain regex patterns for fraudulent posting heuristics
MONEY_PATTERNS = re.compile(
    r"(\$\s*\d+|\b\d+\s*usd\b|wire\s*transfer|western\s*union|moneygram|cashier\s*check|check\s*cashing|payroll\s*check)",
    re.IGNORECASE,
)
CRYPTO_PATTERNS = re.compile(r"\b(bitcoin|crypto|cryptocurrency|usdt|eth|wallet|blockchain\s*payment)\b", re.IGNORECASE)
URGENCY_PATTERNS = re.compile(
    r"\b(urgently\s*hiring|immediate\s*start|immediate\s*hire|act\s*now|act\s*fast|hurry|no\s*experience\s*needed|work\s*from\s*anywhere|earn\s*from\s*home|make\s*money\s*fast)\b",
    re.IGNORECASE,
)
PERSONAL_EMAIL_PATTERNS = re.compile(
    r"[\w\.-]+@(gmail\.com|yahoo\.com|hotmail\.com|outlook\.com|aol\.com|protonmail\.com)",
    re.IGNORECASE,
)
CHAT_APP_PATTERNS = re.compile(
    r"\b(telegram|whatsapp|signal|skype\s*id|skype\s*interview|google\s*hangout)\b",
    re.IGNORECASE,
)
UPFRONT_FEE_PATTERNS = re.compile(
    r"\b(starter\s*kit|registration\s*fee|processing\s*fee|investment\s*required|upfront\s*payment|purchase\s*equipment|refundable\s*deposit|send\s*money)\b",
    re.IGNORECASE,
)


class JobFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts tabular & text-engineered features from job postings DataFrame.
    Outputs a clean numeric DataFrame suitable for tree models and linear models.
    """

    def __init__(self, top_n_categories: int = 15):
        self.top_n_categories = top_n_categories
        self.cat_mappings: Dict[str, Dict[str, float]] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit frequency encodings for categorical columns."""
        df = X.copy()
        for col in ["employment_type", "required_experience", "required_education"]:
            if col in df.columns:
                counts = df[col].fillna("Unknown").value_counts(normalize=True)
                self.cat_mappings[col] = counts.to_dict()
        return self

    def _extract_single_row_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract all numeric features for a single posting."""
        title = str(row.get("title", "") or "")
        desc = str(row.get("description", "") or "")
        profile = str(row.get("company_profile", "") or "")
        reqs = str(row.get("requirements", "") or "")
        benefits = str(row.get("benefits", "") or "")
        combined = f"{title} {profile} {desc} {reqs} {benefits}".strip()

        char_len = float(len(combined))
        words = combined.split()
        word_count = float(len(words))
        avg_word_len = (char_len / word_count) if word_count > 0 else 0.0

        # Character / token distribution
        upper_count = sum(1 for c in combined if c.isupper())
        excl_count = combined.count("!")
        quest_count = combined.count("?")
        digit_count = sum(1 for c in combined if c.isdigit())

        upper_ratio = (upper_count / char_len) if char_len > 0 else 0.0
        excl_ratio = (excl_count / word_count) if word_count > 0 else 0.0
        quest_ratio = (quest_count / word_count) if word_count > 0 else 0.0
        digit_ratio = (digit_count / char_len) if char_len > 0 else 0.0

        # Title specific heuristics
        title_caps = sum(1 for c in title if c.isupper()) / max(1, len(title))
        title_len = float(len(title))

        # Scam regex triggers
        money_matches = float(len(MONEY_PATTERNS.findall(combined)))
        crypto_matches = float(len(CRYPTO_PATTERNS.findall(combined)))
        urgency_matches = float(len(URGENCY_PATTERNS.findall(combined)))
        email_matches = float(len(PERSONAL_EMAIL_PATTERNS.findall(combined)))
        chat_matches = float(len(CHAT_APP_PATTERNS.findall(combined)))
        fee_matches = float(len(UPFRONT_FEE_PATTERNS.findall(combined)))

        total_scam_flags = (
            (1.0 if money_matches > 0 else 0.0)
            + (1.0 if crypto_matches > 0 else 0.0)
            + (1.0 if urgency_matches > 0 else 0.0)
            + (1.0 if email_matches > 0 else 0.0)
            + (1.0 if chat_matches > 0 else 0.0)
            + (1.0 if fee_matches > 0 else 0.0)
        )

        # Presence flags
        has_profile = 1.0 if (len(profile.strip()) > 30 and profile != "Unknown") else 0.0
        has_reqs = 1.0 if len(reqs.strip()) > 10 else 0.0
        has_benefits_flag = 1.0 if len(benefits.strip()) > 5 else 0.0
        has_logo = float(row.get("has_company_logo", 0) or 0)
        has_questions_flag = float(row.get("has_questions", 0) or 0)
        telecommute_flag = float(row.get("telecommuting", 0) or 0)

        # Salary features
        has_salary = float(row.get("has_salary", 0.0) or 0.0)
        salary_spread = float(row.get("salary_spread", 0.0) or 0.0)
        salary_log_min = float(row.get("salary_log_min", 0.0) or 0.0)

        # Categorical frequency mappings
        emp_type = str(row.get("employment_type", "Unknown"))
        req_exp = str(row.get("required_experience", "Unknown"))
        req_edu = str(row.get("required_education", "Unknown"))

        emp_freq = self.cat_mappings.get("employment_type", {}).get(emp_type, 0.0)
        exp_freq = self.cat_mappings.get("required_experience", {}).get(req_exp, 0.0)
        edu_freq = self.cat_mappings.get("required_education", {}).get(req_edu, 0.0)

        return {
            "char_length": char_len,
            "word_count": word_count,
            "avg_word_length": avg_word_len,
            "uppercase_ratio": upper_ratio,
            "exclamation_ratio": excl_ratio,
            "question_ratio": quest_ratio,
            "digit_ratio": digit_ratio,
            "title_length": title_len,
            "title_caps_ratio": title_caps,
            "money_triggers": money_matches,
            "crypto_triggers": crypto_matches,
            "urgency_triggers": urgency_matches,
            "personal_email_triggers": email_matches,
            "chat_app_triggers": chat_matches,
            "upfront_fee_triggers": fee_matches,
            "total_scam_flags": total_scam_flags,
            "has_company_profile": has_profile,
            "has_requirements": has_reqs,
            "has_benefits": has_benefits_flag,
            "has_company_logo": has_logo,
            "has_questions": has_questions_flag,
            "telecommuting": telecommute_flag,
            "has_salary": has_salary,
            "salary_spread": salary_spread,
            "salary_log_min": salary_log_min,
            "employment_type_freq": emp_freq,
            "experience_freq": exp_freq,
            "education_freq": edu_freq,
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into extracted feature matrix."""
        df = X.copy()
        feature_rows = [self._extract_single_row_features(row) for _, row in df.iterrows()]
        features_df = pd.DataFrame(feature_rows, index=df.index)
        self.feature_names_ = list(features_df.columns)
        return features_df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_
