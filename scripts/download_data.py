"""
JobGuard AI - Dataset Downloader
Downloads the EMSCAD 'Real / Fake Job Posting Prediction' dataset (~17,880 postings).
Expected file location: data/raw/fake_job_postings.csv

Usage:
    python scripts/download_data.py [--force] [--use-mirror]
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

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

KAGGLE_DATASET = "shivamb/real-or-fake-fake-jobposting-prediction"
MIRROR_URL = (
    "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv"
)

RAW_DATA_DIR = Path("data/raw")
TARGET_FILE = RAW_DATA_DIR / "fake_job_postings.csv"


def has_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist in environment or default file."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.is_file()


def download_with_kaggle(dest_dir: Path) -> bool:
    """Attempt download using the Kaggle Python API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"[INFO] Authenticated with Kaggle API. Downloading '{KAGGLE_DATASET}'...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(dest_dir), unzip=True)
        return True
    except Exception as e:
        print(f"[WARNING] Kaggle API download failed: {e}")
        return False


def download_from_mirror(dest_file: Path) -> bool:
    """Download verified EMSCAD dataset from public mirror."""
    print(f"[INFO] Downloading EMSCAD dataset from verified mirror: {MIRROR_URL}")
    try:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        # Download with progress report
        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r[INFO] Progress: {percent:5.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MIRROR_URL, dest_file, reporthook=_reporthook)
        print("\n[INFO] Mirror download complete.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Mirror download failed: {e}")
        if dest_file.exists():
            dest_file.unlink()
        return False


def validate_csv(csv_path: Path) -> bool:
    """Validate that the downloaded CSV exists and matches expected schema."""
    if not csv_path.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(csv_path, nrows=5)
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"[ERROR] CSV is missing expected columns: {missing_cols}")
            return False
        total_rows = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
        print(f"[SUCCESS] Dataset validated: {csv_path} contains ~{total_rows:,} records.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to validate CSV: {e}")
        return False


def print_manual_instructions():
    """Print explicit manual download steps for user."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("1. Visit the Kaggle dataset page:")
    print("   https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
    print("2. Click 'Download' to obtain the archive.")
    print("3. Extract 'fake_job_postings.csv' and place it at:")
    print(f"   {TARGET_FILE.resolve()}")
    print("4. Alternatively, configure your Kaggle API key:")
    print("   - Place kaggle.json in ~/.kaggle/kaggle.json, OR")
    print("   - Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download EMSCAD Fake Job Posting Dataset")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    parser.add_argument("--use-mirror", action="store_true", default=True, help="Use verified public mirror if Kaggle credentials absent")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not args.force:
        print(f"[INFO] Target file already exists: {TARGET_FILE}")
        if validate_csv(TARGET_FILE):
            print("[INFO] Existing dataset is valid. Use --force to redownload.")
            return

    download_success = False

    if has_kaggle_credentials():
        print("[INFO] Kaggle credentials found. Attempting download via Kaggle API...")
        download_success = download_with_kaggle(RAW_DATA_DIR)
        if download_success:
            # Check if extracted file needs renaming or is already fake_job_postings.csv
            if not TARGET_FILE.exists():
                for f in RAW_DATA_DIR.glob("*.csv"):
                    if "fake_job" in f.name.lower():
                        f.rename(TARGET_FILE)
                        break

    if not download_success:
        if not has_kaggle_credentials():
            print("[INFO] No Kaggle API credentials detected in environment or ~/.kaggle/kaggle.json.")
            print_manual_instructions()

        if args.use_mirror:
            print("[INFO] Falling back to verified public mirror...")
            download_success = download_from_mirror(TARGET_FILE)

    if TARGET_FILE.exists() and validate_csv(TARGET_FILE):
        print(f"[SUCCESS] Dataset successfully saved to {TARGET_FILE.resolve()}")
    else:
        print(f"[FATAL] Failed to obtain valid dataset at {TARGET_FILE}.")
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
