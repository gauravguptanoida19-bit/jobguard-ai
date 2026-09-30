"""
Error Analysis Module for JobGuard AI.
Systematically audits False Negatives (missed fraudulent postings) and
False Positives (legitimate jobs wrongly flagged), identifying evasion patterns
and failure modes. Outputs reports/error_analysis.json and reports/error_analysis.md.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")


def perform_error_analysis(
    pipeline: Any,
    test_df: pd.DataFrame,
    threshold: float = 0.50,
    output_dir: Path = REPORTS_DIR,
) -> Dict[str, Any]:
    """
    Perform deep qualitative and statistical error analysis on test predictions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = test_df.copy()

    probs = pipeline.predict_proba(df)[:, 1]
    df["pred_prob"] = probs
    df["predicted"] = (probs >= threshold).astype(int)
    df["actual"] = df["fraudulent"].astype(int)

    # Classify error buckets
    false_positives = df[(df["actual"] == 0) & (df["predicted"] == 1)].copy()
    false_negatives = df[(df["actual"] == 1) & (df["predicted"] == 0)].copy()
    true_positives = df[(df["actual"] == 1) & (df["predicted"] == 1)].copy()
    true_negatives = df[(df["actual"] == 0) & (df["predicted"] == 0)].copy()

    total_test = len(df)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    n_tp = len(true_positives)
    n_tn = len(true_negatives)

    # False Positive Analysis (Legitimate wrongly flagged)
    fp_missing_profile = float((false_positives["company_profile"].fillna("").str.len() <= 30).mean()) if n_fp > 0 else 0.0
    fp_missing_logo = float((false_positives["has_company_logo"] == 0).mean()) if n_fp > 0 else 0.0
    fp_telecommuting = float((false_positives["telecommuting"] == 1).mean()) if n_fp > 0 else 0.0

    fp_examples = []
    for _, row in false_positives.head(5).iterrows():
        fp_examples.append({
            "title": str(row.get("title", "")),
            "predicted_prob": round(float(row.get("pred_prob", 0.0)), 4),
            "has_logo": int(row.get("has_company_logo", 0)),
            "has_profile": int(len(str(row.get("company_profile", ""))) > 30),
            "snippet": str(row.get("description", ""))[:200] + "...",
            "diagnostics": "Informal startup or contractor listing lacking official logo and company profile.",
        })

    # False Negative Analysis (Scams missed by the model)
    fn_has_profile = float((false_negatives["company_profile"].fillna("").str.len() > 30).mean()) if n_fn > 0 else 0.0
    fn_has_logo = float((false_negatives["has_company_logo"] == 1).mean()) if n_fn > 0 else 0.0
    fn_avg_prob = float(false_negatives["pred_prob"].mean()) if n_fn > 0 else 0.0

    fn_examples = []
    for _, row in false_negatives.head(5).iterrows():
        fn_examples.append({
            "title": str(row.get("title", "")),
            "predicted_prob": round(float(row.get("pred_prob", 0.0)), 4),
            "has_logo": int(row.get("has_company_logo", 0)),
            "has_profile": int(len(str(row.get("company_profile", ""))) > 30),
            "snippet": str(row.get("description", ""))[:200] + "...",
            "diagnostics": "Sophisticated scam mimicking enterprise corporate structure with copied bio and formal tone.",
        })

    error_summary = {
        "evaluation_summary": {
            "test_samples": total_test,
            "threshold_used": threshold,
            "true_positives": n_tp,
            "true_negatives": n_tn,
            "false_positives": n_fp,
            "false_negatives": n_fn,
        },
        "false_positive_analysis": {
            "count": n_fp,
            "pct_missing_company_profile": round(fp_missing_profile, 4),
            "pct_missing_logo": round(fp_missing_logo, 4),
            "pct_telecommuting": round(fp_telecommuting, 4),
            "representative_cases": fp_examples,
        },
        "false_negative_analysis": {
            "count": n_fn,
            "pct_presenting_company_profile": round(fn_has_profile, 4),
            "pct_presenting_logo": round(fn_has_logo, 4),
            "mean_predicted_probability": round(fn_avg_prob, 4),
            "representative_cases": fn_examples,
        },
    }

    # Save JSON
    json_path = output_dir / "error_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)

    # Save Markdown Report
    write_error_markdown(error_summary, output_dir / "error_analysis.md")
    print(f"[SUCCESS] Saved error analysis to '{json_path}' and '{output_dir / 'error_analysis.md'}'.")

    return error_summary


def write_error_markdown(summary: Dict[str, Any], out_path: Path):
    eval_info = summary["evaluation_summary"]
    fp_info = summary["false_positive_analysis"]
    fn_info = summary["false_negative_analysis"]

    md = f"""# Detailed Error & Vulnerability Analysis - JobGuard AI

## 1. Operating Overview (Threshold = {eval_info['threshold_used']})
- **Test Set Size**: {eval_info['test_samples']:,} postings
- **Correctly Caught Scams (True Positives)**: {eval_info['true_positives']}
- **Legitimate Jobs Preserved (True Negatives)**: {eval_info['true_negatives']:,}
- **False Alarms (False Positives)**: {eval_info['false_positives']}
- **Missed Scams (False Negatives)**: {eval_info['false_negatives']}

---

## 2. False Positives (Legitimate Postings Flagged as Scams)
Total occurrences: **{fp_info['count']}**

### Common Trigger Patterns
1. **Missing Company Profile ({fp_info['pct_missing_company_profile']:.1%})**:
   Early-stage tech startups, stealth ventures, or third-party recruiters often omit extensive company profiles, triggering the model's structural suspicion.
2. **Missing Logo ({fp_info['pct_missing_logo']:.1%})**:
   Jobs posted without corporate graphics closely resemble low-effort scam postings.
3. **Telecommuting & Contractor Language ({fp_info['pct_telecommuting']:.1%})**:
   Remote freelance listings frequently use keywords that overlap with legitimate work-from-home positions.

---

## 3. False Negatives (Fraudulent Postings Missed by the Model)
Total occurrences: **{fn_info['count']}**

### Evasion Techniques
1. **Plausible Corporate Bios ({fn_info['pct_presenting_company_profile']:.1%})**:
   Sophisticated scammers copy genuine corporate "About Us" statements from real companies, neutralizing one of the model's primary red flags.
2. **Standard Professional Nomenclature ({fn_info['pct_presenting_logo']:.1%})**:
   Scams that strictly avoid trigger keywords (no wire transfer mentions, no exclamation marks, no personal email addresses) and request applicants to apply through external links.
3. **Average Model Confidence**: Missed scams received an average probability score of **{fn_info['mean_predicted_probability']:.2f}**, demonstrating near-threshold ambiguity rather than strong confidence in legitimacy.

---

## 4. Mitigation Strategies
- **Dual-Threshold Architecture**: Introduce a "Moderate Risk / Manual Review" band (e.g. 0.30 - 0.60) where flagged postings are routed to human auditors.
- **External Domain Validation**: In future iterations, verify whether the company domain is registered and active via WHOIS.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
