# Data Leakage & Integrity Report - JobGuard AI

## Executive Summary
Data leakage is a critical vulnerability in fraud detection. If near-duplicate job descriptions or company listings cross the train/test boundary, models achieve falsely inflated evaluation scores and degrade in production. Furthermore, spurious correlation with dataset artifacts (e.g. sequential identifiers) must be strictly prevented.

This report documents the systematic checks performed on the EMSCAD dataset (~17,880 postings) to ensure 100% leakage-free evaluation.

---

## 1. Train/Test Boundary Integrity

### Group-Stratified Splitting Methodology
- **Leakage Vector**: Recruiters and scam operators frequently post multiple variants of the same job with minor tweaks (e.g. different locations or departments). Random splitting places identical text signatures in both train and test partitions.
- **Solution**: We implemented `assign_group_clusters` based on MD5 text fingerprints and company profiles, followed by `StratifiedGroupKFold`. All postings sharing identical core descriptions or the same company profile are strictly partitioned to either train or test.

### Split Verification Metrics
- **Train Set Size**: 12,467 postings (534 fraudulent, **4.28%**)
- **Test Set Size**: 3,117 postings (133 fraudulent, **4.27%**)
- **Fingerprint Overlap Between Splits**: **0**
- **Leakage-Free Verified**: **PASSED (Zero Overlap)**

---

## 2. Feature-Level Leakage Checks

### A. Sequential Identifiers (`job_id`)
- **Finding**: `job_id` exhibits an artificial correlation with the label due to dataset collection order.
- **Handling**: `job_id` is **strictly excluded** from all feature engineering, baseline models, and training pipelines.

### B. Missingness Signals vs. Leakage
- **Company Profile**: 67.3% of fraudulent postings lack a company profile, compared to only 15.1% of legitimate postings.
- **Benefits**: ~75% of fraudulent postings omit benefits vs 43% in legitimate listings.
- **Verdict**: This represents genuine behavioral fraud signals (scammers rarely take time to draft detailed company bios or structured benefits), not collection leakage. We encode these as explicit binary presence indicators (`has_company_profile`, `has_benefits`).

### C. Company Logo (`has_company_logo`)
- Only 35.8% of fraudulent postings include a company logo, compared to 82.3% of legitimate postings.
- This is a legitimate structural feature reflecting low-effort scam postings.

---

## 3. Near-Duplicate Deduplication
- Across the raw dataset, deduplication identified and isolated duplicate postings.
- The training and test splits retain strict group segregation, ensuring test evaluation reflects unseen postings.
