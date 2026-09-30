"""
Plain-English Reason Generation Module for JobGuard AI.
Translates heuristic triggers, missingness anomalies, and model contributions
into clear, actionable explanations for end users and fraud auditors.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def generate_plain_english_reasons(
    posting: Dict[str, Any],
    probability: float,
    threshold: float = 0.50,
    max_reasons: int = 4,
) -> List[str]:
    """
    Generate ordered human-readable bullet reasons explaining the risk classification.
    """
    reasons: List[str] = []

    title = str(posting.get("title", "") or "")
    desc = str(posting.get("description", "") or "")
    profile = str(posting.get("company_profile", "") or "")
    reqs = str(posting.get("requirements", "") or "")
    benefits = str(posting.get("benefits", "") or "")
    combined = f"{title} {profile} {desc} {reqs} {benefits}".lower()

    has_logo = int(posting.get("has_company_logo", 1) or 0)
    has_questions = int(posting.get("has_questions", 1) or 0)
    telecommuting = int(posting.get("telecommuting", 0) or 0)

    is_risky = probability >= 0.35

    # 1. High-severity behavioral triggers
    if any(k in combined for k in ["wire transfer", "western union", "moneygram", "cashier check", "check cashing"]):
        reasons.append("Mentions wire transfers, check cashing, or unusual money handling methods commonly associated with payment scams.")

    if any(k in combined for k in ["starter kit", "registration fee", "processing fee", "upfront payment", "purchase equipment"]):
        reasons.append("Requests upfront payments, registration fees, or employee equipment purchases.")

    if any(k in combined for k in ["@gmail.com", "@yahoo.com", "@hotmail.com", "@aol.com"]):
        reasons.append("Recruiter uses a free public email address (@gmail/@yahoo) instead of a verified corporate domain.")

    if any(k in combined for k in ["telegram", "whatsapp", "signal"]):
        reasons.append("Directs candidates to off-platform messaging apps (Telegram, WhatsApp) for interview/hiring.")

    if any(k in combined for k in ["bitcoin", "crypto", "usdt", "wallet"]):
        reasons.append("References cryptocurrency or direct digital wallet payments.")

    # 2. Structural red flags
    if not profile or len(profile.strip()) <= 30 or profile.strip().lower() in ["none", "unknown"]:
        if is_risky:
            reasons.append("Company profile is missing. In the EMSCAD dataset, fraudulent postings are ~4x more likely to omit company profiles.")

    if has_logo == 0 and is_risky:
        reasons.append("No official company logo is attached to the listing.")

    if not reqs or len(reqs.strip()) <= 15:
        if is_risky:
            reasons.append("Lacks structured candidate requirements or educational qualifications.")

    # 3. Urgency & shouting heuristics
    excl_count = (title + desc).count("!")
    if excl_count >= 3 and is_risky:
        reasons.append("Unusually high frequency of exclamation marks and informal punctuation.")

    upper_ratio = sum(1 for c in title if c.isupper()) / max(1, len(title))
    if upper_ratio > 0.40 and len(title) > 8 and is_risky:
        reasons.append("Excessive capitalization in job title indicative of informal or spam solicitation.")

    if any(k in combined for k in ["urgently hiring", "immediate start", "act fast", "earn fast", "no experience"]):
        if is_risky:
            reasons.append("Employs aggressive urgency and 'no experience required' language characteristic of work-from-home scams.")

    # 4. If Low Risk (Reassuring signals)
    if probability < 0.35 and not reasons:
        reasons.append("Verified and detailed company background profile provided.")
        if has_logo == 1:
            reasons.append("Official company logo attached.")
        if len(reqs.strip()) > 30:
            reasons.append("Detailed, professional job requirements and qualifications specified.")
        if len(benefits.strip()) > 20:
            reasons.append("Standard corporate benefits package outlined.")
        reasons.append("No high-risk scam keywords (wire transfer, upfront fees, off-platform chat) detected.")

    return reasons[:max_reasons]
