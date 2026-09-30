# Detailed Error & Vulnerability Analysis - JobGuard AI

## 1. Operating Overview (Threshold = 0.65)
- **Test Set Size**: 3,117 postings
- **Correctly Caught Scams (True Positives)**: 71
- **Legitimate Jobs Preserved (True Negatives)**: 2,956
- **False Alarms (False Positives)**: 28
- **Missed Scams (False Negatives)**: 62

---

## 2. False Positives (Legitimate Postings Flagged as Scams)
Total occurrences: **28**

### Common Trigger Patterns
1. **Missing Company Profile (75.0%)**:
   Early-stage tech startups, stealth ventures, or third-party recruiters often omit extensive company profiles, triggering the model's structural suspicion.
2. **Missing Logo (64.3%)**:
   Jobs posted without corporate graphics closely resemble low-effort scam postings.
3. **Telecommuting & Contractor Language (0.0%)**:
   Remote freelance listings frequently use keywords that overlap with legitimate work-from-home positions.

---

## 3. False Negatives (Fraudulent Postings Missed by the Model)
Total occurrences: **62**

### Evasion Techniques
1. **Plausible Corporate Bios (51.6%)**:
   Sophisticated scammers copy genuine corporate "About Us" statements from real companies, neutralizing one of the model's primary red flags.
2. **Standard Professional Nomenclature (54.8%)**:
   Scams that strictly avoid trigger keywords (no wire transfer mentions, no exclamation marks, no personal email addresses) and request applicants to apply through external links.
3. **Average Model Confidence**: Missed scams received an average probability score of **0.32**, demonstrating near-threshold ambiguity rather than strong confidence in legitimacy.

---

## 4. Mitigation Strategies
- **Dual-Threshold Architecture**: Introduce a "Moderate Risk / Manual Review" band (e.g. 0.30 - 0.60) where flagged postings are routed to human auditors.
- **External Domain Validation**: In future iterations, verify whether the company domain is registered and active via WHOIS.
