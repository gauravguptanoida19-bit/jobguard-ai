"""
Data cleaning and preprocessing module for JobGuard AI.
Handles text normalization, HTML stripping, salary parsing, missing values, and deduplication.
"""

import re
import html
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


def strip_html_and_clean_text(text: Optional[str]) -> str:
    """
    Remove HTML tags, decode HTML entities, and normalize whitespace.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Decode HTML entities like &amp;, &lt;, etc.
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs if needed or normalize
    text = re.sub(r"http[s]?://\S+", " [URL] ", text)
    # Normalize excessive whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_salary_range(salary_val: Optional[str]) -> Dict[str, float]:
    """
    Parse the EMSCAD 'salary_range' column (e.g., '10000-20000', '0-0', 'NaN').
    
    Returns:
        dict with keys:
            - 'has_salary': 1.0 if valid non-zero range, else 0.0
            - 'salary_min': minimum salary or 0.0
            - 'salary_max': maximum salary or 0.0
            - 'salary_spread': difference or 0.0
            - 'salary_log_min': log1p of salary_min
    """
    default_res = {
        "has_salary": 0.0,
        "salary_min": 0.0,
        "salary_max": 0.0,
        "salary_spread": 0.0,
        "salary_log_min": 0.0,
    }

    if pd.isna(salary_val) or not str(salary_val).strip():
        return default_res

    salary_str = str(salary_val).strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", salary_str)
    if not match:
        return default_res

    try:
        val_min = float(match.group(1))
        val_max = float(match.group(2))
    except (ValueError, OverflowError):
        return default_res

    # Filter out placeholder ranges like 0-0
    if val_min == 0.0 and val_max == 0.0:
        return default_res

    # Ensure min <= max
    if val_min > val_max:
        val_min, val_max = val_max, val_min

    spread = val_max - val_min
    log_min = float(np.log1p(val_min))

    return {
        "has_salary": 1.0,
        "salary_min": val_min,
        "salary_max": val_max,
        "salary_spread": spread,
        "salary_log_min": log_min,
    }


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical columns to standard sets, filling NaNs with 'Unknown'."""
    df = df.copy()
    cat_columns = [
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
    ]
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    # Clean boolean flags
    flag_columns = ["telecommuting", "has_company_logo", "has_questions"]
    for col in flag_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


def clean_job_postings(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    create_combined_text: bool = True,
) -> pd.DataFrame:
    """
    Perform full data cleaning on EMSCAD job postings DataFrame.
    
    Steps:
    1. Deduplicate by job posting text content (if drop_duplicates=True)
    2. Clean individual text fields (title, company_profile, description, requirements, benefits)
    3. Generate missingness indicator flags
    4. Parse salary_range into numeric features
    5. Normalize categorical fields
    6. Construct unified 'combined_text'
    """
    df = df.copy()

    # Text columns in EMSCAD
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]

    # 1. Indicator flags for presence of key fields BEFORE cleaning to empty
    for col in text_cols:
        if col in df.columns:
            df[f"has_{col}"] = df[col].notna().astype(int) & (df[col].astype(str).str.strip().str.len() > 0).astype(int)
        else:
            df[col] = ""
            df[f"has_{col}"] = 0

    # 2. Clean individual text columns
    for col in text_cols:
        df[col] = df[col].apply(strip_html_and_clean_text)

    # 3. Deduplication based on title + description
    if drop_duplicates:
        initial_len = len(df)
        df["_dedup_hash"] = df["title"].str.lower() + "___" + df["description"].str.lower()
        # Drop duplicates, keeping the first
        df = df.drop_duplicates(subset=["_dedup_hash"]).drop(columns=["_dedup_hash"])
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"[INFO] Deduplication: dropped {dropped} duplicate postings ({len(df)} remaining).")

    # 4. Parse salary range
    if "salary_range" in df.columns:
        salary_feats = df["salary_range"].apply(parse_salary_range).apply(pd.Series)
        for c in salary_feats.columns:
            df[c] = salary_feats[c]
    else:
        df["has_salary"] = 0.0
        df["salary_min"] = 0.0
        df["salary_max"] = 0.0
        df["salary_spread"] = 0.0
        df["salary_log_min"] = 0.0

    # 5. Normalize categoricals
    df = normalize_categoricals(df)

    # 6. Combined text for NLP representations
    if create_combined_text:
        # Create structured combined text
        df["combined_text"] = (
            "Job Title: " + df["title"]
            + "\nCompany Profile: " + df["company_profile"]
            + "\nDescription: " + df["description"]
            + "\nRequirements: " + df["requirements"]
            + "\nBenefits: " + df["benefits"]
        ).str.strip()

    return df
