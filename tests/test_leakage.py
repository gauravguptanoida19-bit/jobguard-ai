"""
Unit tests for data leakage prevention and group-stratified splitting.
"""

import pandas as pd
import numpy as np
import pytest

from src.features.leakage import (
    compute_content_fingerprint,
    assign_group_clusters,
    create_leakage_free_split,
    check_leakage_between_splits,
)


def test_content_fingerprint():
    t1 = "Senior Python Developer"
    d1 = "We are seeking a senior python engineer to build backend APIs."
    t2 = "senior python developer"
    d2 = "we are seeking a senior python engineer to build backend apis."
    
    fp1 = compute_content_fingerprint(t1, d1)
    fp2 = compute_content_fingerprint(t2, d2)
    assert fp1 == fp2

    # Different content
    t3 = "Graphic Designer"
    d3 = "Design logos and marketing materials using Photoshop."
    fp3 = compute_content_fingerprint(t3, d3)
    assert fp1 != fp3


def test_leakage_free_split_no_overlap():
    # Construct a dataset with duplicate postings from same company
    records = []
    # Company A has 4 identical postings (legitimate)
    for i in range(4):
        records.append({
            "title": "Accountant",
            "description": "Handle monthly reconciliations and taxes.",
            "company_profile": "Acme Financial Services Corporation",
            "fraudulent": 0,
        })
    # Company B has 3 duplicate scam postings
    for i in range(3):
        records.append({
            "title": "Online Assistant",
            "description": "Work from home and receive checks weekly.",
            "company_profile": "",
            "fraudulent": 1,
        })
    # Add other distinct postings
    for i in range(20):
        records.append({
            "title": f"Engineer {i}",
            "description": f"Unique engineering job description for position {i}",
            "company_profile": f"Company {i}",
            "fraudulent": 1 if i < 3 else 0,
        })

    df = pd.DataFrame(records)
    train_df, test_df = create_leakage_free_split(df, test_size=0.25, random_state=42)

    # Check that company A is either 100% in train OR 100% in test
    comp_a_train = (train_df["company_profile"] == "Acme Financial Services Corporation").sum()
    comp_a_test = (test_df["company_profile"] == "Acme Financial Services Corporation").sum()
    assert (comp_a_train == 0 and comp_a_test == 4) or (comp_a_train == 4 and comp_a_test == 0)

    # Run leakage checker
    leakage_result = check_leakage_between_splits(train_df, test_df)
    assert leakage_result["is_leakage_free"] is True
    assert leakage_result["fingerprint_overlap_count"] == 0


def test_check_leakage_detects_contamination():
    # Artificially create overlapping splits
    train_df = pd.DataFrame([{
        "title": "Data Scientist",
        "description": "Work on deep learning and predictive models.",
        "fraudulent": 0,
    }])
    test_df = pd.DataFrame([{
        "title": "Data Scientist",
        "description": "Work on deep learning and predictive models.",
        "fraudulent": 0,
    }])

    res = check_leakage_between_splits(train_df, test_df)
    assert res["is_leakage_free"] is False
    assert res["fingerprint_overlap_count"] == 1
