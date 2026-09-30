"""
Data Leakage Prevention and Detection Module for JobGuard AI.
Ensures near-duplicates do not cross train/test splits, checks for target-leaking fields,
and guarantees stratified split integrity.
"""

import hashlib
import re
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def compute_content_fingerprint(title: str, description: str) -> str:
    """
    Compute a normalized text fingerprint for near-duplicate detection.
    Normalizes whitespace and lowercase to catch near-identical postings.
    """
    t = " ".join(str(title).lower().split())
    d = " ".join(str(description).lower().split())
    # Match on title and prefix of description
    content = f"{t}___{d[:250]}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class UnionFind:
    """Disjoint Set Union (Union-Find) for clustering overlapping entities."""

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        if self.parent.setdefault(item, item) != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b


def assign_group_clusters(df: pd.DataFrame) -> pd.Series:
    """
    Assign group cluster IDs using Union-Find on content fingerprints and company profiles.
    Guarantees that postings with matching content OR shared company profiles form
    mutually exclusive connected components.
    """
    uf = UnionFind()
    row_fps = []

    for i, (_, row) in enumerate(df.iterrows()):
        row_id = f"row_{i}"
        title = str(row.get("title", ""))
        desc = str(row.get("description", ""))
        profile = str(row.get("company_profile", "")).strip()

        # Connect row to its content fingerprint
        fp = f"fp_{compute_content_fingerprint(title, desc)}"
        uf.union(row_id, fp)

        # Connect row to company profile if substantial
        if profile and len(profile) > 30 and profile != "Unknown":
            comp_hash = f"comp_{hashlib.md5(profile.lower().encode('utf-8')).hexdigest()}"
            uf.union(row_id, comp_hash)

        row_fps.append(row_id)

    # Assign root canonical ID to each row
    group_ids = [uf.find(r) for r in row_fps]
    return pd.Series(group_ids, index=df.index, name="leakage_group_id")


def create_leakage_free_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets such that:
    1. Near-duplicate postings and company profiles do not cross the train/test boundary.
    2. Class proportions (fraudulent rate) are preserved as closely as possible.
    """
    df = df.copy()
    if "leakage_group_id" not in df.columns:
        df["leakage_group_id"] = assign_group_clusters(df)

    y = df["fraudulent"].values
    groups = df["leakage_group_id"].values
    unique_groups = len(np.unique(groups))

    # Approximate n_splits = int(1.0 / test_size)
    desired_splits = max(2, int(round(1.0 / test_size)))
    n_splits = min(desired_splits, unique_groups)

    if n_splits < 2 or unique_groups <= 2:
        # Fall back to standard StratifiedKFold if groups cannot be partitioned
        skf = StratifiedKFold(n_splits=max(2, desired_splits), shuffle=True, random_state=random_state)
        train_idx, test_idx = next(skf.split(df, y))
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(sgkf.split(df, y, groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Verification: check group intersection
    train_groups = set(train_df["leakage_group_id"])
    test_groups = set(test_df["leakage_group_id"])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Data leakage detected! {len(overlap)} groups exist in both train and test splits.")

    return train_df, test_df


def check_leakage_between_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Verify zero content leakage between train and test splits.
    """
    train_prints: Set[str] = set()
    for _, row in train_df.iterrows():
        train_prints.add(compute_content_fingerprint(str(row.get("title", "")), str(row.get("description", ""))))

    test_prints: Set[str] = set()
    for _, row in test_df.iterrows():
        test_prints.add(compute_content_fingerprint(str(row.get("title", "")), str(row.get("description", ""))))

    leaked = train_prints.intersection(test_prints)
    train_fraud_rate = float(train_df["fraudulent"].mean())
    test_fraud_rate = float(test_df["fraudulent"].mean())

    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_fraud_count": int(train_df["fraudulent"].sum()),
        "test_fraud_count": int(test_df["fraudulent"].sum()),
        "train_fraud_rate": round(train_fraud_rate, 4),
        "test_fraud_rate": round(test_fraud_rate, 4),
        "fingerprint_overlap_count": len(leaked),
        "is_leakage_free": len(leaked) == 0,
    }


def analyze_feature_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze individual dataset features to detect potential target leakage:
    - High mutual information or 100% correlation with target
    - Columns with IDs or artifact numbering
    - Rare categories appearing exclusively in fraudulent records
    """
    leakage_findings = []
    
    # 1. Check job_id correlation
    if "job_id" in df.columns:
        corr = df["job_id"].corr(df["fraudulent"])
        leakage_findings.append({
            "feature": "job_id",
            "type": "sequential_identifier",
            "correlation_with_target": round(float(corr), 4),
            "recommendation": "EXCLUDE: job_id is an arbitrary sequential index and must not be used as a predictor."
        })

    # 2. Check categorical value exclusivity
    cat_cols = ["employment_type", "required_experience", "required_education", "telecommuting", "has_company_logo", "has_questions"]
    for col in cat_cols:
        if col in df.columns:
            # Check fraud rate per category
            group_stats = df.groupby(col)["fraudulent"].agg(["count", "mean"]).reset_index()
            for _, row in group_stats.iterrows():
                val = row[col]
                count = row["count"]
                fraud_rate = row["mean"]
                if count >= 10 and fraud_rate >= 0.95:
                    leakage_findings.append({
                        "feature": col,
                        "value": str(val),
                        "count": int(count),
                        "fraud_rate": round(float(fraud_rate), 4),
                        "recommendation": "WARNING: Potential target leakage or strong deterministic subset. Apply smoothing or regularization."
                    })

    # 3. Missingness correlation
    for col in ["company_profile", "requirements", "benefits", "salary_range"]:
        if col in df.columns:
            missing_rate_real = float(df[df["fraudulent"] == 0][col].isna().mean())
            missing_rate_fake = float(df[df["fraudulent"] == 1][col].isna().mean())
            leakage_findings.append({
                "feature": f"missing_{col}",
                "missing_rate_real": round(missing_rate_real, 4),
                "missing_rate_fake": round(missing_rate_fake, 4),
                "diff": round(missing_rate_fake - missing_rate_real, 4),
                "recommendation": (
                    f"Informative feature: fake postings have {missing_rate_fake:.1%} missing vs {missing_rate_real:.1%} in real. "
                    "Legitimate signal rather than leakage, but must be represented as boolean flag without overfitting."
                )
            })

    return {
        "findings": leakage_findings,
        "total_records": len(df),
        "total_fraudulent": int(df["fraudulent"].sum()),
        "baseline_fraud_rate": round(float(df["fraudulent"].mean()), 4),
    }
