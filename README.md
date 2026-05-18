# AuditAI Analytics Platform

**Enterprise-grade audit data analytics for Big 4 external audit engagements.**

Combines Benford's Law statistical testing, ML anomaly detection, and journal entry risk scoring to identify fraud indicators, manipulation patterns, and material misstatement risks — aligned with ISA 240, ISA 315, ISA 500.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

To pre-generate the sample dataset (10,000 GL entries):
```bash
python data/generate_sample_data.py
```

---

## Features

### 1. Benford's Law Engine
Tests whether the leading digit distribution of general ledger amounts conforms to Benford's Law — a powerful indicator of fabricated or manipulated entries.

- **1st digit, 2nd digit, first-two digit analysis**
- **Chi-square test** with p-value and critical value
- **Mean Absolute Deviation (MAD)** — Nigrini (2012) conformity thresholds
- **Z-score per digit** (|Z| > 1.96 flagged at 5% significance)
- **Kolmogorov-Smirnov test** as additional evidence
- **Segment analysis** by entity, account type, or user
- ISA 240 risk commentary auto-generated

**Theoretical basis:** Benford's Law states that in naturally occurring numerical datasets, the leading digit d appears with probability P(d) = log₁₀(1 + 1/d). Deviations indicate potential manipulation, particularly "just below threshold" amounts, repeated invented figures, or systematic rounding. Reference: Nigrini, M.J. (2012). *Benford's Law.* Wiley.

### 2. Journal Entry Risk Scoring (0–100)
Each journal entry is scored across 12 risk indicators aligned with ISA 240.A3:

| Indicator | Weight | ISA Reference |
|-----------|--------|---------------|
| Same-day reversal | 25 | ISA 240.A3 |
| Round number (multiples of 1k, 5k, 10k…) | 20 | ISA 240.A3 / Nigrini |
| Suspense / clearing account | 20 | ISA 240.A33 |
| Manual override | 15 | ISA 240.A3 |
| Period-end entry (last N days) | 15 | ISA 240.A3 |
| Unusual posting hour (after 20:00 / before 06:00) | 15 | ISA 240.A3 |
| Low-activity user | 15 | ISA 240.A3 |
| Weekend posting | 10 | ISA 240.A3 |
| Public holiday posting | 10 | ISA 240.A3 |
| Large amount (> 5× materiality) | 10 | ISA 315.A107 |
| Intercompany entry | 5 | ISA 315.A85 |
| Invoice splitting (just below threshold) | 15 | ISA 240.A3 / FCPA |

Risk tiers: **Critical (70–100)** · **High (40–70)** · **Medium (15–40)** · **Low (0–15)**

### 3. ML Anomaly Detection
Multivariate anomaly detection using **Isolation Forest** (Liu et al., 2008).

Features: log-amount, day of month, day of week, posting hour, month, weekend flag, month-end flag, account type, user profile, debit/credit indicator.

- **SHAP values** (if shap library available): explains the top risk driver per entry
- **K-Means clustering** of anomalies into behavioural groups
- **Overlap analysis** between ML and statistical flags (consensus entries = highest priority)

### 4. Vendor / Counterparty Analysis
- **Duplicate payment detection** — same vendor, same amount, within configurable window
- **Ghost vendor detection** — fuzzy name matching (FuzzyWuzzy) to identify near-duplicate vendor names
- **Concentration analysis** — top-N vendors by spend, cumulative %
- **Payment timing analysis** — by day of week and time window

### 5. Audit Dashboard (Streamlit)
Six-tab professional interface:

| Tab | Content |
|-----|---------|
| Executive Summary | Overall risk, KPIs, timeline, top indicators |
| Benford's Law | Distribution charts, Z-score, segment analysis |
| Journal Entry Risk | Scatter map, indicator breakdown, filterable table |
| ML Anomaly Detection | Scatter plots, top anomalies, SHAP drivers |
| Vendor Analysis | Duplicates, ghost vendors, concentration, timing |
| Audit Report | Auto-generated memo, export (Excel / TXT / CSV) |

### 6. Audit Report Generator
Professional audit memorandum:
- **ISA 240, ISA 315, ISA 500, PCAOB AS 2401** references
- Findings ranked by risk level (Critical / High / Medium / Low)
- Excel workbook export: JE Exceptions, Benford Analysis, ML Anomalies, Duplicate Payments, Ghost Vendors
- Text memo suitable for inclusion in audit working papers

---

## Sample Dataset

The sample General Ledger (`data/sample_general_ledger.csv`) contains **10,000 journal entries** across 5 fictitious entities with **10 embedded fraud / manipulation scenarios**:

| Scenario | Description |
|----------|-------------|
| F-01 | Round numbers clustered at $9,800–$9,999 (just below $10k threshold) |
| F-02 | Late-night / weekend postings by low-activity user SYS_TEMP_01 |
| F-03 | Duplicate payments to Apex Consulting (same invoice ref) |
| F-04 | Ghost vendor "Apexe Consulting" (92% name similarity to Apex) |
| F-05 | Year-end revenue pull-forward (Dec 28–31, inflated amounts) |
| F-06 | Benford violation in TechFlow Inc vendor payments (excess digits 7/8) |
| F-07 | Same-day reversals of large expense entries |
| F-08 | Large-value routing through Clearing / Suspense account 9100 |
| F-09 | Invoice splitting — 5 × $9,500 instead of 1 × $47,500 |
| F-10 | Intercompany entries at quarter-end |

---

## ISA Standards Alignment

| Standard | Application |
|----------|------------|
| **ISA 240** | Fraud risk factors, journal entry testing, Benford's Law risk commentary |
| **ISA 315** | Understanding the entity, identifying risks of material misstatement |
| **ISA 500** | Audit evidence — analytical procedures as substantive tests |
| **ISA 530** | Audit sampling — using risk scores to stratify the population |
| **ISA 560** | Subsequent events — period-end entry analysis |
| **PCAOB AS 2401** | Consideration of fraud in a financial statement audit |

---

## Configurable Parameters (Streamlit Sidebar)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Engagement materiality | $10,000 | Amount threshold for large-entry flag and split detection |
| Period-end window | 3 days | Days before month/quarter/year-end to flag entries |
| Statistical significance α | 5% | Chi-square and Z-test significance level |
| Digit type | 1st digit | Benford digit analysis (1st / 2nd / first-two) |
| Benford min amount | $1 | Exclude trivial amounts from Benford population |
| Anomaly rate | 5% | Isolation Forest contamination parameter |
| IF trees | 200 | Number of Isolation Forest estimators |
| Anomaly clusters | 4 | K-Means cluster count for anomaly grouping |
| Off-hours start | 20:00 | Posting hour after which entries are flagged |
| Off-hours end | 06:00 | Posting hour before which entries are flagged |
| Duplicate window | 30 days | Same-vendor same-amount lookback window |
| Ghost vendor threshold | 85% | Fuzzy name similarity score to flag as ghost vendor |

---

## Project Structure

```
AUDIT/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── README.md
├── src/
│   ├── benford_analyzer.py         # Benford's Law engine (chi², MAD, Z-score, KS)
│   ├── anomaly_detector.py         # Isolation Forest + SHAP explanations
│   ├── journal_entry_analyzer.py   # 12-indicator JE risk scoring
│   ├── vendor_analyzer.py          # Duplicate, ghost vendor, concentration
│   └── risk_reporter.py            # Audit memo + Excel export
├── data/
│   ├── generate_sample_data.py     # 10,000-row GL generator with fraud scenarios
│   └── sample_general_ledger.csv   # Generated sample (auto-created on first run)
└── tests/
    ├── test_benford.py
    └── test_anomaly.py
```

---

## References

- Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection.* Wiley.
- Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). *Isolation Forest.* ICDM.
- IAASB (2009). *ISA 240 — The Auditor's Responsibilities Relating to Fraud.*
- IAASB (2019). *ISA 315 (Revised) — Identifying and Assessing the Risks of Material Misstatement.*
- IAASB (2009). *ISA 500 — Audit Evidence.*
- PCAOB (2010). *AS 2401 — Consideration of Fraud in a Financial Statement Audit.*
- COSO (2016). *Fraud Risk Management Guide.*
- Lundberg, S. & Lee, S.I. (2017). *A unified approach to interpreting model predictions (SHAP).* NeurIPS.
