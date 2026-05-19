# AuditAI Analytics Platform

**Professional audit data analytics platform for external audit engagements.**

Combines Benford's Law statistical testing, journal entry risk scoring, ML anomaly detection, vendor analysis, AI-generated audit memos, and an integrated audit assistant chatbot — aligned with ISA 240, ISA 315, ISA 500, and PCAOB AS 2401.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

To pre-generate the sample dataset (~10,000 GL entries):
```bash
python data/generate_sample_data.py
```

---

## Features

### 1. Benford's Law Engine
Tests whether the leading digit distribution of general ledger amounts conforms to Benford's Law — a powerful indicator of fabricated or manipulated entries.

- 1st digit, 2nd digit, first-two digit analysis
- Chi-square test with p-value and critical value
- Mean Absolute Deviation (MAD) with Nigrini (2012) conformity thresholds
- Z-score per digit (|Z| > 1.96 flagged at 5% significance)
- Kolmogorov-Smirnov test as supplementary evidence
- Segment analysis by entity, account type, or user
- ISA 240 risk commentary auto-generated

**Theoretical basis:** Benford's Law states that in naturally occurring numerical datasets, the leading digit d appears with probability P(d) = log10(1 + 1/d). Deviations indicate potential manipulation, particularly "just below threshold" amounts, repeated invented figures, or systematic rounding. Reference: Nigrini, M.J. (2012). *Benford's Law.* Wiley.

### 2. Journal Entry Risk Scoring (0-100)
Each journal entry is scored across 12 risk indicators aligned with ISA 240.A3:

| Indicator | Weight | ISA Reference |
|-----------|--------|---------------|
| Same-day reversal | 25 | ISA 240.A3 |
| Round number (multiples of 1k, 5k, 10k...) | 20 | ISA 240.A3 / Nigrini |
| Suspense / clearing account | 20 | ISA 240.A33 |
| Manual override | 15 | ISA 240.A3 |
| Period-end entry (last N days of period) | 15 | ISA 240.A3 |
| Unusual posting hour (after 20:00 / before 06:00) | 15 | ISA 240.A3 |
| Low-activity user | 15 | ISA 240.A3 |
| Weekend posting | 10 | ISA 240.A3 |
| Public holiday posting | 10 | ISA 240.A3 |
| Large amount (> 5x materiality) | 10 | ISA 315.A107 |
| Intercompany entry | 5 | ISA 315.A85 |
| Invoice splitting (just below threshold) | 15 | ISA 240.A3 / FCPA |

Risk tiers: **Critical (70-100)** · **High (40-70)** · **Medium (15-40)** · **Low (0-15)**

### 3. ML Anomaly Detection
Multivariate anomaly detection using Isolation Forest (Liu et al., 2008).

Features: log-amount, day of month, day of week, posting hour, month, weekend flag, month-end flag, account type, user profile, debit/credit indicator.

- SHAP values (optional): explains the top risk driver per anomalous entry
- K-Means clustering of anomalies into behavioural groups
- Overlap analysis between ML and statistical flags — consensus entries = highest audit priority

### 4. Vendor / Counterparty Analysis
- Duplicate payment detection: same vendor, same amount, within configurable window
- Ghost vendor detection: fuzzy name matching (FuzzyWuzzy) to identify near-duplicate vendor names
- Concentration analysis: top-N vendors by spend with cumulative percentage
- Payment timing analysis by day of week and time window

### 5. AI Report Generation (LLM Integration)
Generates a complete professional audit memorandum (Working Paper DA-001) from the analytics results using a Large Language Model of your choice:

| Provider | Models | Notes |
|----------|--------|-------|
| **Groq (recommended)** | llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it | Fastest, free tier available |
| Google Gemini | gemini-2.0-flash, gemini-1.5-pro | Fast |
| OpenAI | gpt-4o, gpt-4-turbo, gpt-3.5-turbo | Standard |
| Anthropic Claude | claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5 | High quality |

The generated memo includes: Purpose and Scope, Executive Risk Assessment, Benford findings, JE testing results, ML anomaly summary, vendor risk, ranked findings with recommended procedures, audit strategy impact, limitations, and conclusion — all with specific ISA paragraph references.

**API key configuration:** Enter directly in the sidebar, or pre-configure in Streamlit secrets:
```toml
# .streamlit/secrets.toml
[api_keys]
groq      = "gsk_..."
gemini    = "AIza..."
openai    = "sk-..."
anthropic = "sk-ant-..."
```

### 6. AuditAI Assistant Chatbot (LLM Integration)
An interactive audit assistant powered by the same LLM providers. The chatbot has expert knowledge of:

- Benford's Law (theory, statistical tests, MAD thresholds, Z-score interpretation)
- Journal entry testing (ISA 240.A3 indicators, fraud risk factors)
- ML anomaly detection (how Isolation Forest works, SHAP values, contamination parameter)
- Vendor analysis (ghost vendor schemes, duplicate payment detection, fuzzy matching)
- ISA standards (ISA 240, 315, 500, 530, 560, PCAOB AS 2401)
- Platform parameters (how to configure materiality, alpha, contamination, etc.)
- Interpreting results (what risk levels mean, what to do with a HIGH Benford risk)

Responds in English or French depending on the question language. Includes 10 bilingual starter questions for first-time users.

### 7. Audit Dashboard (Streamlit)
Seven-tab professional interface:

| Tab | Content |
|-----|---------|
| Executive Summary | Overall risk, KPIs, timeline, top indicators |
| Benford's Law | Distribution charts, Z-score per digit, segment analysis |
| Journal Entry Risk | Scatter map, indicator breakdown, filterable exceptions table |
| ML Anomaly Detection | Scatter plots, top anomalies, SHAP risk drivers, cluster analysis |
| Vendor Analysis | Duplicates, ghost vendors, concentration chart, payment timing |
| Audit Report | AI memo generation + export (Excel workbook / TXT / CSV) |
| Assistant | Interactive chatbot for audit concepts and platform guidance |

---

## Sample Dataset

The sample General Ledger (`data/sample_general_ledger.csv`) contains approximately 10,000 journal entries across 5 fictitious entities with 10 embedded fraud and manipulation scenarios:

| Scenario | Description |
|----------|-------------|
| F-01 | Round numbers clustered at $9,800-$9,999 (just below $10k approval threshold) |
| F-02 | Late-night / weekend postings by low-activity user SYS_TEMP_01 |
| F-03 | Duplicate payments to Apex Consulting (same invoice reference) |
| F-04 | Ghost vendor "Apexe Consulting" (92% name similarity to Apex Consulting) |
| F-05 | Year-end revenue pull-forward (Dec 28-31, inflated amounts) |
| F-06 | Benford violation in TechFlow Inc vendor payments (excess digits 7 and 8) |
| F-07 | Same-day reversals of large expense entries |
| F-08 | Large-value routing through Clearing / Suspense account 9100 |
| F-09 | Invoice splitting: 5 x $9,500 instead of 1 x $47,500 |
| F-10 | Intercompany entries at quarter-end |

---

## ISA Standards Alignment

| Standard | Application |
|----------|------------|
| **ISA 240** | Fraud risk factors, journal entry testing, Benford's Law risk commentary |
| **ISA 315** | Understanding the entity, identifying risks of material misstatement |
| **ISA 500** | Audit evidence: analytical procedures as substantive tests |
| **ISA 530** | Audit sampling: using risk scores to stratify the population |
| **ISA 560** | Subsequent events: period-end entry analysis |
| **PCAOB AS 2401** | Consideration of fraud in a financial statement audit |

---

## Configurable Parameters (Streamlit Sidebar)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Engagement materiality | $10,000 | Amount threshold for large-entry flag and split detection |
| Period-end window | 3 days | Days before month/quarter/year-end to flag entries |
| Statistical significance alpha | 5% | Chi-square and Z-test significance level |
| Digit type | 1st digit | Benford digit analysis (1st / 2nd / first-two) |
| Benford min amount | $1 | Exclude trivial amounts from Benford population |
| Anomaly rate | 5% | Isolation Forest contamination parameter |
| IF trees | 200 | Number of Isolation Forest estimators |
| Anomaly clusters | 4 | K-Means cluster count for anomaly grouping |
| Off-hours start | 20:00 | Posting hour after which entries are flagged |
| Off-hours end | 06:00 | Posting hour before which entries are flagged |
| Public holiday country | US | ISO 3166-1 alpha-2 code (FR, GB, DE, CA, AU, BE...) |
| Duplicate window | 30 days | Same-vendor same-amount lookback window |
| Ghost vendor threshold | 85% | Fuzzy name similarity score to flag as ghost vendor |
| LLM Provider | Groq | Provider for AI memo generation and the chatbot |

---

## Project Structure

```
AUDIT/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── README.md
├── src/
│   ├── benford_analyzer.py         # Benford's Law engine (chi-square, MAD, Z-score, KS)
│   ├── anomaly_detector.py         # Isolation Forest + SHAP + K-Means clustering
│   ├── journal_entry_analyzer.py   # 12-indicator JE risk scoring (ISA 240.A3)
│   ├── vendor_analyzer.py          # Duplicate, ghost vendor, concentration analysis
│   ├── risk_reporter.py            # Text memo + Excel multi-sheet export
│   ├── ai_reporter.py              # AI audit memo generation (4 LLM providers)
│   └── audit_chatbot.py            # AuditAI Assistant chatbot (4 LLM providers)
├── data/
│   ├── generate_sample_data.py     # GL generator with 10 fraud scenarios
│   └── sample_general_ledger.csv   # Auto-generated on first run
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
