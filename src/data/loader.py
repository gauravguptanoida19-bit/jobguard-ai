"""
Data loading module for JobGuard AI.
Strictly requires real data from the EMSCAD dataset.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

EXPECTED_COLUMNS = [
    "job_id",
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
]

DEFAULT_RAW_PATH = Path("data/raw/fake_job_postings.csv")


def load_raw_data(data_path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load raw EMSCAD fake job postings CSV.
    
    Raises:
        FileNotFoundError: If the CSV file is missing, giving helpful instructions.
        ValueError: If essential columns are missing or file is corrupted.
    """
    path = Path(data_path) if data_path else DEFAULT_RAW_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"\n[DATASET MISSING] Could not find dataset at: '{path.resolve()}'.\n"
            f"Please run the download script:\n"
            f"    python scripts/download_data.py\n"
            f"Or manually download 'fake_job_postings.csv' from:\n"
            f"    https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
            f"and place it at 'data/raw/fake_job_postings.csv'.\n"
            f"Note: JobGuard AI enforces strict ML engineering practices and does not allow synthetic data."
        )

    df = pd.read_csv(path)

    # Check for target column
    if "fraudulent" not in df.columns:
        raise ValueError(f"Dataset at '{path}' is missing the required target column 'fraudulent'.")

    return df
