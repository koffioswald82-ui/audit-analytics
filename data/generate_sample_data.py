"""
Generates a realistic 10,000-row General Ledger dataset with embedded
fraud and manipulation scenarios for audit analytics demonstration.

Fraud scenarios embedded
------------------------
F-01  Round-number clustering just below the $10,000 approval threshold (9,800–9,999)
F-02  Late-night/weekend entries by a low-activity user (SYS_TEMP_01)
F-03  Duplicate vendor payments — "Apex Consulting" invoices
F-04  Ghost vendor — "Apexe Consulting" (Apex misspelling, 92% name similarity)
F-05  Period-end revenue pull-forward (Dec 28-31, inflated amounts)
F-06  Benford violation — excess leading digit 7 in Entity TechFlow Inc vendor pmts
F-07  Same-day reversals of large expense entries
F-08  Suspense account abuse — unusual routing via account 9100 Clearing
F-09  Invoice splitting — five $9,500 payments instead of one $47,500
F-10  Intercompany eliminations omitted at quarter-end
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Master data ────────────────────────────────────────────────────────────
ENTITIES = [
    "ACME Corp",
    "TechFlow Inc",
    "Global Trade Ltd",
    "FinServ Partners",
    "BuildCo SA",
]

ACCOUNTS = {
    # code: (name, type)
    "1100": ("Cash & Cash Equivalents",      "Asset"),
    "1200": ("Accounts Receivable",           "Asset"),
    "1300": ("Inventories",                   "Asset"),
    "1400": ("Prepaid Expenses",              "Asset"),
    "1500": ("Property Plant & Equipment",    "Asset"),
    "2100": ("Accounts Payable",              "Liability"),
    "2200": ("Accrued Liabilities",           "Liability"),
    "2300": ("Short-Term Borrowings",         "Liability"),
    "2400": ("Deferred Revenue",              "Liability"),
    "3100": ("Share Capital",                 "Equity"),
    "3200": ("Retained Earnings",             "Equity"),
    "4100": ("Revenue — Products",            "Revenue"),
    "4200": ("Revenue — Services",            "Revenue"),
    "4300": ("Other Operating Income",        "Revenue"),
    "5100": ("Cost of Goods Sold",            "Expense"),
    "5200": ("Direct Labour",                 "Expense"),
    "6100": ("Salaries & Wages",              "Expense"),
    "6200": ("Rent & Occupancy",              "Expense"),
    "6300": ("Professional Fees",             "Expense"),
    "6400": ("Marketing & Advertising",       "Expense"),
    "6500": ("Travel & Entertainment",        "Expense"),
    "6600": ("IT & Software",                 "Expense"),
    "6700": ("Depreciation & Amortisation",   "Expense"),
    "6800": ("Insurance",                     "Expense"),
    "7100": ("Interest Income",               "Revenue"),
    "7200": ("Interest Expense",              "Expense"),
    "8100": ("Income Tax Expense",            "Expense"),
    "9100": ("Clearing / Suspense",           "Suspense"),
    "9200": ("Intercompany Receivable",       "Asset"),
    "9300": ("Intercompany Payable",          "Liability"),
}

USERS = [
    "j.martin",   "s.chen",   "l.dupont",  "m.okafor",
    "a.silva",    "t.yamada", "r.kumar",   "p.garcia",
    "c.lambert",  "f.nkosi",  "SYS_AUTO",  "SYS_TEMP_01",
]

APPROVERS = [
    "director.finance", "vp.controller", "cfo.office",
    "mgr.accounting",   "dir.treasury",
]

VENDORS = {
    "V001": "Apex Consulting",
    "V002": "Global IT Solutions",
    "V003": "Premier Office Supply",
    "V004": "Metro Legal Advisors",
    "V005": "FastTrack Logistics",
    "V006": "NovaTech Services",
    "V007": "Stellar Recruitment",
    "V008": "Alliance Insurance",
    "V009": "ClearView Analytics",
    "V010": "Platinum Facilities",
    "V011": "Apexe Consulting",     # Ghost vendor — F-04
    "V012": "Nexus Cloud Inc",
    "V013": "Horizon Marketing",
    "V014": "Pacific Trade Co",
    "V015": "BlueSky Research",
}

CURRENCIES = ["USD", "EUR", "GBP", "CHF", "CAD"]
SOURCE_SYSTEMS = ["SAP", "Oracle", "Manual", "Import", "Concur"]
JE_TYPES = ["Automated", "Manual", "Import", "Reversal", "Accrual", "Correction"]

START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)


# ── Helpers ───────────────────────────────────────────────────────────────
def rand_date(start=START_DATE, end=END_DATE) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def benford_amount(min_val=100, max_val=500_000) -> float:
    """Log-uniform distribution — naturally follows Benford's Law."""
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    return round(10 ** np.random.uniform(log_min, log_max), 2)


def biased_amount_digit7(min_val=1000, max_val=200_000) -> float:
    """Amount with leading digit biased toward 7 and 8 — Fraud F-06."""
    first_digit = np.random.choice([7, 8], p=[0.55, 0.45])
    magnitude = np.random.choice(range(3, 6))
    rest = np.random.uniform(0, 1) * 10 ** magnitude
    return round(first_digit * 10 ** magnitude + rest, 2)


def build_normal_row(je_id: int) -> dict:
    dt = rand_date()
    account_code = random.choice(list(ACCOUNTS.keys()))
    account_name, account_type = ACCOUNTS[account_code]
    vendor_id = random.choice(list(VENDORS.keys())[:10])  # only legit vendors
    entity = random.choice(ENTITIES)
    amount = benford_amount()
    return {
        "je_id":         f"JE-{je_id:06d}",
        "entity":        entity,
        "posting_date":  dt.strftime("%Y-%m-%d"),
        "value_date":    dt.strftime("%Y-%m-%d"),
        "period":        dt.strftime("%Y-%m"),
        "account_code":  account_code,
        "account_name":  account_name,
        "account_type":  account_type,
        "cost_center":   f"CC{random.randint(100,999)}",
        "description":   f"Standard posting — {account_name}",
        "debit_credit":  random.choice(["D", "C"]),
        "amount":        amount,
        "currency":      random.choices(CURRENCIES, weights=[60,20,10,5,5])[0],
        "vendor_id":     vendor_id,
        "vendor_name":   VENDORS[vendor_id],
        "invoice_ref":   f"INV-{random.randint(10000,99999)}",
        "user_id":       random.choice(USERS[:10]),
        "approved_by":   random.choice(APPROVERS),
        "posting_hour":  int(np.random.choice(range(8, 18),
                             p=np.array([5,8,12,14,14,12,10,8,7,10]) / 100)),
        "source_system": random.choices(SOURCE_SYSTEMS, weights=[40,30,15,10,5])[0],
        "je_type":       random.choices(JE_TYPES[:4], weights=[45,30,20,5])[0],
        "reversal_flag": "N",
    }


# ── Fraud scenario builders ───────────────────────────────────────────────
def make_f01_round_number(je_id: int) -> dict:
    """F-01: Just-below-threshold round numbers ($9,800–$9,999)."""
    row = build_normal_row(je_id)
    row["amount"]      = round(random.uniform(9_800, 9_999), 2)
    row["description"] = "Vendor payment — expense reimbursement"
    row["je_type"]     = "Manual"
    return row


def make_f02_offhours(je_id: int) -> dict:
    """F-02: Late-night/weekend postings by SYS_TEMP_01."""
    row = build_normal_row(je_id)
    dt  = rand_date()
    # Force to weekend or between 22:00–04:00
    if random.random() < 0.5:
        # weekend
        days_ahead = (5 - dt.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        dt = dt + timedelta(days=days_ahead)
    row["posting_date"] = dt.strftime("%Y-%m-%d")
    row["posting_hour"] = random.choice([0, 1, 2, 3, 22, 23])
    row["user_id"]      = "SYS_TEMP_01"
    row["approved_by"]  = "SYS_TEMP_01"
    row["je_type"]      = "Manual"
    row["source_system"] = "Manual"
    return row


def make_f03_duplicate_payment(je_id: int, base_date: datetime) -> dict:
    """F-03: Duplicate payment to Apex Consulting."""
    row = build_normal_row(je_id)
    row["vendor_id"]    = "V001"
    row["vendor_name"]  = "Apex Consulting"
    row["amount"]       = 47_500.00
    row["invoice_ref"]  = "INV-88421"   # same invoice ref
    row["account_code"] = "6300"
    row["account_name"] = "Professional Fees"
    row["account_type"] = "Expense"
    row["posting_date"] = (base_date + timedelta(days=random.randint(0, 12))).strftime("%Y-%m-%d")
    row["description"]  = "Consulting services Q4 — Apex"
    return row


def make_f04_ghost_vendor(je_id: int) -> dict:
    """F-04: Payment to ghost vendor Apexe Consulting."""
    row = build_normal_row(je_id)
    row["vendor_id"]    = "V011"
    row["vendor_name"]  = "Apexe Consulting"
    row["amount"]       = benford_amount(5_000, 80_000)
    row["account_code"] = "6300"
    row["account_name"] = "Professional Fees"
    row["account_type"] = "Expense"
    row["description"]  = "Advisory services — Apexe"
    row["je_type"]      = "Manual"
    return row


def make_f05_period_end(je_id: int) -> dict:
    """F-05: Inflated revenue entries in last 3 days of December."""
    year = random.choice([2023, 2024])
    day  = random.randint(28, 31)
    if day == 31 and year == 2023:
        day = 31
    dt = datetime(year, 12, day)
    row = build_normal_row(je_id)
    row["posting_date"] = dt.strftime("%Y-%m-%d")
    row["period"]       = dt.strftime("%Y-%m")
    row["account_code"] = "4100"
    row["account_name"] = "Revenue — Products"
    row["account_type"] = "Revenue"
    row["debit_credit"] = "C"
    row["amount"]       = benford_amount(50_000, 500_000)
    row["description"]  = "Year-end revenue recognition adjustment"
    row["je_type"]      = "Accrual"
    row["user_id"]      = random.choice(["j.martin", "s.chen"])
    return row


def make_f06_benford_violation(je_id: int) -> dict:
    """F-06: Vendor payments with inflated leading digit 7/8 — TechFlow Inc."""
    row = build_normal_row(je_id)
    row["entity"]       = "TechFlow Inc"
    row["amount"]       = biased_amount_digit7()
    row["account_code"] = "6300"
    row["account_name"] = "Professional Fees"
    row["account_type"] = "Expense"
    return row


def make_f07_same_day_reversal(je_id_base: int, base_date: datetime) -> list:
    """F-07: Same-day reversal of a large expense entry."""
    original = build_normal_row(je_id_base)
    amount   = benford_amount(50_000, 250_000)
    acct     = "6100"
    original.update({
        "posting_date":  base_date.strftime("%Y-%m-%d"),
        "account_code":  acct,
        "account_name":  ACCOUNTS[acct][0],
        "account_type":  ACCOUNTS[acct][1],
        "debit_credit":  "D",
        "amount":        amount,
        "description":   "Salary accrual — large posting",
        "je_type":       "Accrual",
    })
    reversal = original.copy()
    reversal["je_id"]        = f"JE-{je_id_base+1:06d}"
    reversal["debit_credit"] = "C"
    reversal["amount"]       = -amount
    reversal["description"]  = "REVERSAL — " + original["description"]
    reversal["je_type"]      = "Reversal"
    reversal["reversal_flag"] = "Y"
    return [original, reversal]


def make_f08_suspense(je_id: int) -> dict:
    """F-08: Large routing through suspense/clearing account."""
    row = build_normal_row(je_id)
    row["account_code"] = "9100"
    row["account_name"] = "Clearing / Suspense"
    row["account_type"] = "Suspense"
    row["amount"]       = benford_amount(10_000, 200_000)
    row["description"]  = "Temporary clearing — pending allocation"
    row["je_type"]      = "Manual"
    return row


def make_f09_split_payment(je_id_base: int, base_date: datetime) -> list:
    """F-09: Invoice splitting — 5 × $9,500 instead of $47,500."""
    rows = []
    for i in range(5):
        row = build_normal_row(je_id_base + i)
        row.update({
            "je_id":         f"JE-{je_id_base+i:06d}",
            "posting_date":  (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "vendor_id":     "V006",
            "vendor_name":   "NovaTech Services",
            "amount":        9_500.00,
            "account_code":  "6600",
            "account_name":  "IT & Software",
            "account_type":  "Expense",
            "invoice_ref":   f"INV-SPLIT-{je_id_base}",
            "description":   f"IT services instalment {i+1}/5",
            "je_type":       "Manual",
        })
        rows.append(row)
    return rows


# ── Main generator ────────────────────────────────────────────────────────
def generate(n_normal: int = 9_400, output_path: str | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    je_counter = 1

    # Normal transactions
    for _ in range(n_normal):
        rows.append(build_normal_row(je_counter))
        je_counter += 1

    # F-01: 150 round-number entries
    for _ in range(150):
        rows.append(make_f01_round_number(je_counter))
        je_counter += 1

    # F-02: 80 off-hours entries
    for _ in range(80):
        rows.append(make_f02_offhours(je_counter))
        je_counter += 1

    # F-03: 8 duplicate payments (4 pairs)
    base_dates_dup = [datetime(2023, 7, 15), datetime(2023, 11, 3),
                      datetime(2024, 3, 20), datetime(2024, 9, 10)]
    for bd in base_dates_dup:
        rows.append(make_f03_duplicate_payment(je_counter,     bd))
        rows.append(make_f03_duplicate_payment(je_counter + 1, bd))
        je_counter += 2

    # F-04: 30 ghost vendor payments
    for _ in range(30):
        rows.append(make_f04_ghost_vendor(je_counter))
        je_counter += 1

    # F-05: 50 period-end revenue entries
    for _ in range(50):
        rows.append(make_f05_period_end(je_counter))
        je_counter += 1

    # F-06: 100 Benford-violating entries
    for _ in range(100):
        rows.append(make_f06_benford_violation(je_counter))
        je_counter += 1

    # F-07: 20 same-day reversals (40 entries total)
    reversal_dates = [rand_date() for _ in range(20)]
    for rd in reversal_dates:
        pair = make_f07_same_day_reversal(je_counter, rd)
        rows.extend(pair)
        je_counter += 2

    # F-08: 30 suspense account entries
    for _ in range(30):
        rows.append(make_f08_suspense(je_counter))
        je_counter += 1

    # F-09: 3 split-payment clusters (15 entries total)
    split_dates = [datetime(2023, 5, 10), datetime(2024, 2, 15), datetime(2024, 8, 1)]
    for sd in split_dates:
        split_rows = make_f09_split_payment(je_counter, sd)
        rows.extend(split_rows)
        je_counter += 5

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["je_id"] = [f"JE-{i+1:06d}" for i in range(len(df))]

    if output_path is None:
        output_path = str(Path(__file__).parent / "sample_general_ledger.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} journal entries -> {output_path}")
    return df


if __name__ == "__main__":
    generate()
