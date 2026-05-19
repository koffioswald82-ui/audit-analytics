"""
AuditAI Analytics Platform
Enterprise-grade audit data analytics dashboard for Big 4 use cases.

Covers: Benford's Law, Journal Entry Risk Scoring, ML Anomaly Detection,
        Vendor Analysis, and automated Audit Memo generation.

Standards: ISA 240 | ISA 315 | ISA 500 | PCAOB AS 2401
"""
import sys
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AuditAI Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:audit-analytics@firm.com",
        "About": "AuditAI Analytics Platform v2.0",
    },
)

# ── Source path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.benford_analyzer import BenfordAnalyzer
from src.anomaly_detector import AnomalyDetector
from src.journal_entry_analyzer import JournalEntryAnalyzer
from src.vendor_analyzer import VendorAnalyzer
from src.risk_reporter import RiskReporter
from src.ai_reporter import AIReporter, PROVIDERS
from src.audit_chatbot import call_chatbot, STARTER_QUESTIONS

# Custom CSS: professional dark-gold theme
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
    border-right: 2px solid #FFE600;
}
[data-testid="stSidebar"] * { color: #E8E8E8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFE600 !important; font-size: 0.85rem !important; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stSidebar"] hr { border-color: #333 !important; }

/* ── Header banner ── */
.ey-banner {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 100%);
    padding: 1.2rem 2rem;
    border-radius: 8px;
    border-left: 6px solid #FFE600;
    margin-bottom: 1.5rem;
}
.ey-banner h1 { color: #FFE600; font-size: 1.7rem; font-weight: 700; margin: 0; }
.ey-banner p  { color: #B0B0C0; font-size: 0.85rem; margin: 0.3rem 0 0; }

/* ── Metric cards ── */
.metric-card {
    background: #FFFFFF;
    border-left: 5px solid #FFE600;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.metric-card .value { font-size: 1.8rem; font-weight: 800; color: #1A1A2E; line-height: 1.2; }
.metric-card .delta { font-size: 0.8rem; color: #666; }

/* ── Risk badges ── */
.badge-critical { background:#DC2626; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.75rem; }
.badge-high     { background:#EA580C; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.75rem; }
.badge-medium   { background:#D97706; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.75rem; }
.badge-low      { background:#059669; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.75rem; }

/* ── Section header ── */
.section-header {
    border-left: 4px solid #FFE600;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
    color: #1A1A2E;
    font-weight: 700;
    font-size: 1.05rem;
}

/* ── Tab bar ── */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #FFE600; }
[data-baseweb="tab"] {
    background: #F5F5F5;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-weight: 600;
    color: #555;
}
[aria-selected="true"] {
    background: #FFE600 !important;
    color: #1A1A2E !important;
}

/* ── Info box ── */
.finding-box {
    background: #FFF9E6;
    border: 1px solid #FFE600;
    border-left: 5px solid #D97706;
    padding: 0.9rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}

/* ── Dataframe override ── */
.stDataFrame { border: 1px solid #E5E5E5; border-radius: 6px; }

/* ── Divider ── */
.gold-divider { height: 2px; background: linear-gradient(90deg, #FFE600, transparent); margin: 1.5rem 0; border: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating sample GL dataset (10,000 entries)...")
def load_sample_data() -> pd.DataFrame:
    sample_path = ROOT / "data" / "sample_general_ledger.csv"
    if not sample_path.exists():
        # Generate on the fly
        gen_spec = importlib.util.spec_from_file_location(
            "generate_sample_data",
            ROOT / "data" / "generate_sample_data.py",
        )
        gen_mod = importlib.util.module_from_spec(gen_spec)
        gen_spec.loader.exec_module(gen_mod)
        df = gen_mod.generate(output_path=str(sample_path))
    else:
        df = pd.read_csv(sample_path, low_memory=False)
    return _parse_df(df)


def _parse_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    if "posting_hour" not in df.columns:
        df["posting_hour"] = 9
    df["posting_hour"] = pd.to_numeric(df["posting_hour"], errors="coerce").fillna(9).astype(int)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df


@st.cache_data(show_spinner="Running Benford analysis...")
def run_benford(_df, digit_type, min_amount, alpha):
    analyzer = BenfordAnalyzer()
    return analyzer.analyze(
        _df["amount"],
        digit_type=digit_type,
        min_amount=min_amount,
        alpha=alpha,
    )


@st.cache_data(show_spinner="Scoring journal entries...")
def run_je_scoring(_df, materiality, period_end_days, unusual_start, unusual_end, country="US"):
    analyzer = JournalEntryAnalyzer(
        materiality=materiality,
        period_end_days=period_end_days,
        unusual_hour_start=unusual_start,
        unusual_hour_end=unusual_end,
        country=country,
    )
    return analyzer.score_entries(_df)


@st.cache_data(show_spinner="Training ML anomaly detector...")
def run_ml_detection(_df, contamination, n_estimators):
    detector = AnomalyDetector(
        contamination=contamination,
        n_estimators=n_estimators,
    )
    detector.fit(_df)
    return detector.predict(_df)


def _risk_color(level: str) -> str:
    return {
        "Critical": "#DC2626",
        "High":     "#EA580C",
        "Medium":   "#D97706",
        "Low":      "#059669",
    }.get(str(level), "#6B7280")


def _fmt_amount(v: float) -> str:
    return f"${v:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar():
    with st.sidebar:
        st.markdown("### AuditAI Analytics")
        st.markdown("*Audit Data Analytics Platform*")
        st.divider()

        # ── Engagement parameters ──────────────────────────────────────
        st.markdown("#### Engagement")
        client_name = st.text_input("Client name", value="Demo Client")
        partner     = st.text_input("Engagement partner", value="Partner A")
        audit_period = st.text_input("Audit period", value="FY 2024")

        st.divider()

        # ── Data source ────────────────────────────────────────────────
        st.markdown("#### Data Source")
        data_source = st.radio(
            "Select data source",
            ["Sample GL (10,000 entries)", "Upload CSV / Excel"],
            index=0,
        )

        uploaded_file = None
        if data_source == "Upload CSV / Excel":
            uploaded_file = st.file_uploader(
                "Upload General Ledger",
                type=["csv", "xlsx", "xls", "xlsm"],
                help="Accepted: CSV, Excel (.xlsx/.xls/.xlsm). See the column guide below.",
            )
            if uploaded_file:
                st.caption(f"File: {uploaded_file.name}")

        st.divider()

        # ── Filters ────────────────────────────────────────────────────
        st.markdown("#### Filters")

        # Populate options from previously loaded data (available on second+ reruns)
        _fdf = st.session_state.get("_raw_df")
        _entity_opts  = sorted(_fdf["entity"].dropna().unique().tolist())      if _fdf is not None and "entity"       in _fdf.columns else []
        _acct_opts    = sorted(_fdf["account_type"].dropna().unique().tolist()) if _fdf is not None and "account_type" in _fdf.columns else []
        _user_opts    = sorted(_fdf["user_id"].dropna().unique().tolist())      if _fdf is not None and "user_id"      in _fdf.columns else []

        filter_entity    = st.multiselect("Entity / company", options=_entity_opts, placeholder="All entities")
        filter_acct_type = st.multiselect("Account type",     options=_acct_opts,  placeholder="All types")
        filter_user      = st.multiselect("Posted by (user)", options=_user_opts,  placeholder="All users")
        filter_dc        = st.radio("Debit / Credit", ["Both", "Debit only", "Credit only"])

        date_range = st.date_input(
            "Posting date range",
            value=(pd.Timestamp("2023-01-01").date(), pd.Timestamp("2024-12-31").date()),
        )

        amount_min = st.number_input("Min amount ($)", value=0.0, step=100.0, format="%.0f")
        amount_max = st.number_input("Max amount ($)", value=0.0, step=10000.0, format="%.0f",
                                     help="Set to 0 for no upper limit")

        st.divider()

        # ── Audit parameters ───────────────────────────────────────────
        st.markdown("#### Audit Parameters")
        materiality = st.number_input(
            "Engagement materiality ($)",
            value=10_000.0, min_value=100.0, step=1_000.0, format="%.0f",
            help="Used for large-amount flagging and invoice-split detection.",
        )
        perf_mat = materiality * 0.75
        st.caption(f"Performance materiality: ${perf_mat:,.0f} (75%)")

        period_end_days = st.slider(
            "Period-end window (days)",
            min_value=1, max_value=7, value=3,
            help="Entries posted in the last N days of a period are flagged.",
        )
        alpha = st.select_slider(
            "Statistical significance (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            help="Significance level for chi-square and Z-tests.",
        )

        st.divider()

        # ── Benford parameters ─────────────────────────────────────────
        st.markdown("#### Benford's Law")
        digit_type = st.selectbox(
            "Digit analysis",
            ["first", "second", "first_two"],
            format_func=lambda x: {"first": "1st Digit", "second": "2nd Digit",
                                    "first_two": "First Two Digits"}[x],
        )
        benford_min_amount = st.number_input(
            "Minimum amount for Benford ($)",
            value=1.0, min_value=0.01, step=1.0, format="%.2f",
            help="Exclude small/trivial amounts from Benford analysis.",
        )
        segment_by = st.selectbox(
            "Segment analysis by",
            ["entity", "account_type", "user_id", "None"],
            help="Run Benford separately per segment.",
        )

        st.divider()

        # ── ML parameters ──────────────────────────────────────────────
        st.markdown("#### ML / AI Parameters")
        contamination = st.slider(
            "Expected anomaly rate (%)",
            min_value=1, max_value=15, value=5,
            help="Isolation Forest contamination parameter: estimated % of anomalies.",
        ) / 100.0
        n_estimators = st.slider(
            "Number of trees (IF)",
            min_value=50, max_value=300, value=200, step=50,
            help="More trees = higher accuracy but slower.",
        )
        n_clusters = st.slider(
            "Anomaly clusters (k-means)",
            min_value=2, max_value=8, value=4,
            help="Number of clusters when grouping anomalies.",
        )

        st.divider()

        # ── Posting hours ──────────────────────────────────────────────
        st.markdown("#### Off-Hours Definition")
        col1, col2 = st.columns(2)
        with col1:
            unusual_start = st.number_input("After (hh)", value=20, min_value=16, max_value=23)
        with col2:
            unusual_end = st.number_input("Before (hh)", value=6, min_value=0, max_value=8)
        holiday_country = st.text_input(
            "Public holiday country (ISO code)",
            value="US",
            max_chars=3,
            help="ISO 3166-1 alpha-2 code. Examples: US, FR, GB, DE, CA, AU, BE, CH",
        ).strip().upper() or "US"

        st.divider()

        # ── Vendor parameters ──────────────────────────────────────────
        st.markdown("#### Vendor Analysis")
        dup_window = st.slider(
            "Duplicate detection window (days)",
            min_value=7, max_value=90, value=30,
            help="Flag same-vendor same-amount entries within this window.",
        )
        fuzzy_threshold = st.slider(
            "Ghost vendor similarity threshold (%)",
            min_value=70, max_value=98, value=85,
            help="Fuzzy name match score to flag as potential ghost vendor.",
        )

        # ── AI Report Generation ───────────────────────────────────────
        st.divider()
        st.markdown("#### AI Report Generation")
        ai_provider = st.selectbox(
            "LLM Provider",
            options=list(PROVIDERS.keys()),
            format_func=lambda k: PROVIDERS[k]["label"],
            index=0,
            help="Groq is the fastest and has a generous free tier.",
        )
        ai_model = st.selectbox(
            "Model",
            options=PROVIDERS[ai_provider]["models"],
        )
        # Read from Streamlit secrets if pre-configured, else sidebar input
        _secret_key = ""
        try:
            _secret_key = st.secrets.get("api_keys", {}).get(ai_provider, "")
        except Exception:
            pass

        if _secret_key:
            st.success("API key loaded from Streamlit secrets.")
            ai_api_key = _secret_key
        else:
            ai_api_key = st.text_input(
                "API Key",
                type="password",
                placeholder=PROVIDERS[ai_provider]["key_hint"],
                help=f"Get your key: {PROVIDERS[ai_provider]['docs']}",
            )
            st.caption(
                f"[Get {PROVIDERS[ai_provider]['label'].split('(')[0].strip()} API key]"
                f"({PROVIDERS[ai_provider]['docs']})"
            )

        st.divider()
        st.caption("AuditAI Analytics v2.0")
        st.caption("ISA 240  |  ISA 315  |  ISA 500")

    return {
        "client_name":      client_name,
        "partner":          partner,
        "audit_period":     audit_period,
        "data_source":      data_source,
        "uploaded_file":    uploaded_file,
        "filter_entity":    filter_entity,
        "filter_acct_type": filter_acct_type,
        "filter_user":      filter_user,
        "filter_dc":        filter_dc,
        "date_range":       date_range,
        "amount_min":       amount_min,
        "amount_max":       amount_max,
        "materiality":      materiality,
        "period_end_days":  period_end_days,
        "alpha":            alpha,
        "digit_type":       digit_type,
        "benford_min":      benford_min_amount,
        "segment_by":       segment_by,
        "contamination":    contamination,
        "n_estimators":     n_estimators,
        "n_clusters":       n_clusters,
        "unusual_start":    int(unusual_start),
        "unusual_end":      int(unusual_end),
        "holiday_country":  holiday_country,
        "dup_window":       dup_window,
        "fuzzy_threshold":  fuzzy_threshold,
        "ai_provider":      ai_provider,
        "ai_model":         ai_model,
        "ai_api_key":       ai_api_key,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Column mapping constants & helpers
# ─────────────────────────────────────────────────────────────────────────────

# Standard column → list of known aliases (case-insensitive)
_COL_ALIASES = {
    "posting_date":  ["date", "post_date", "trans_date", "transaction_date",
                      "date_comptable", "buchungsdatum", "datum", "value_date",
                      "doc_date", "document_date", "gl_date", "period_date"],
    "amount":        ["montant", "amount_lcy", "amount_local", "betrag", "solde",
                      "net_amount", "gross_amount", "value", "sum", "debit_amount",
                      "credit_amount", "transaction_amount", "amt"],
    "entity":        ["company", "company_code", "bukrs", "societe", "entity_code",
                      "legal_entity", "business_unit", "company_name"],
    "account_code":  ["account", "gl_account", "compte", "account_number", "saknr",
                      "hkont", "gl_code", "ledger_account", "cost_account"],
    "account_name":  ["account_description", "account_desc", "gl_description",
                      "compte_libelle", "account_text"],
    "account_type":  ["acc_type", "type_compte", "account_category", "account_class"],
    "user_id":       ["user", "posted_by", "preparer", "created_by", "bname",
                      "entered_by", "entered_user", "posting_user", "utilisateur"],
    "approved_by":   ["approver", "authorized_by", "manager", "supervisor"],
    "vendor_name":   ["vendor", "supplier", "fournisseur", "payee", "creditor",
                      "vendor_description", "supplier_name"],
    "vendor_id":     ["vendor_code", "supplier_code", "lifnr", "payee_id"],
    "description":   ["text", "narration", "memo", "sgtxt", "libelle", "note",
                      "transaction_description", "detail", "reference_text"],
    "debit_credit":  ["dc_indicator", "d_c", "shkzg", "dr_cr", "type"],
    "posting_hour":  ["hour", "heure", "time_hour", "entry_hour"],
    "je_type":       ["entry_type", "journal_type", "source"],
    "period":        ["fiscal_period", "periode", "accounting_period", "fiscal_month"],
    "currency":      ["waers", "curr", "currency_code", "doc_currency"],
    "invoice_ref":   ["invoice", "invoice_number", "ref", "reference", "doc_number"],
    "cost_center":   ["cost_ctr", "profit_center", "cc", "kostl"],
}

_REQUIRED_COLS  = ["posting_date", "amount"]
_RECOMMENDED_COLS = ["entity", "account_code", "account_type", "user_id",
                     "description", "vendor_name", "posting_hour", "debit_credit"]
_OPTIONAL_COLS  = ["approved_by", "vendor_id", "je_type", "period",
                   "currency", "invoice_ref", "cost_center"]

# Schema shown to users
SCHEMA_TABLE = [
    # (standard_name, type, status, description, example)
    ("posting_date",  "Date",    "Required",    "Journal entry posting date",          "2024-03-31"),
    ("amount",        "Numeric", "Required",    "Transaction amount (positive or neg.)", "125,000.00"),
    ("entity",        "Text",    "Recommended", "Legal entity / company",              "ACME Corp"),
    ("account_code",  "Text",    "Recommended", "GL account code",                     "6300"),
    ("account_name",  "Text",    "Recommended", "Account description",                 "Professional Fees"),
    ("account_type",  "Text",    "Recommended", "Asset / Liability / Revenue / Expense","Expense"),
    ("user_id",       "Text",    "Recommended", "User who posted the entry",           "j.martin"),
    ("description",   "Text",    "Recommended", "Transaction narration / memo",        "Consulting fees Q1"),
    ("vendor_name",   "Text",    "Recommended", "Vendor / supplier name",             "Apex Consulting"),
    ("posting_hour",  "Integer", "Recommended", "Hour the entry was posted (0–23)",    "22"),
    ("debit_credit",  "Text",    "Recommended", "D = Debit / C = Credit",             "D"),
    ("approved_by",   "Text",    "Optional",    "Approver name or ID",                 "cfo.office"),
    ("vendor_id",     "Text",    "Optional",    "Vendor / supplier code",             "V001"),
    ("je_type",       "Text",    "Optional",    "Manual / Automated / Reversal",      "Manual"),
    ("period",        "Text",    "Optional",    "Fiscal period (YYYY-MM)",            "2024-03"),
    ("currency",      "Text",    "Optional",    "ISO currency code",                  "USD"),
    ("invoice_ref",   "Text",    "Optional",    "Invoice or document reference",      "INV-88421"),
    ("cost_center",   "Text",    "Optional",    "Cost centre / profit centre",        "CC101"),
]


def _auto_map(df_cols: list) -> dict:
    """Return {standard_col: detected_df_col} using alias matching (case-insensitive)."""
    cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df_cols}
    mapping = {}
    for std_col, aliases in _COL_ALIASES.items():
        # Exact match first
        if std_col in cols_lower:
            mapping[std_col] = cols_lower[std_col]
            continue
        # Alias match
        for alias in aliases:
            if alias in cols_lower:
                mapping[std_col] = cols_lower[alias]
                break
    return mapping


def render_data_guide_and_mapping(uploaded_file) -> dict:
    """
    Renders the column structure guide and interactive column mapper.
    Returns the final column-mapping dict {standard_col: file_col}.
    """
    # ── Schema guide ───────────────────────────────────────────────────
    with st.expander("Expected Column Structure (click to expand)", expanded=not bool(uploaded_file)):
        st.markdown(
            "Your GL file must include **posting_date** and **amount**. "
            "The richer the data, the more analysis modules will activate."
        )
        schema_df = pd.DataFrame(
            SCHEMA_TABLE,
            columns=["Column Name", "Type", "Status", "Description", "Example"],
        )

        def _style_status(row):
            color = {"Required": "#FEE2E2", "Recommended": "#FEF3C7",
                     "Optional": "#F0FDF4"}.get(row["Status"], "")
            return [f"background-color:{color}"] * len(row)

        st.dataframe(
            schema_df.style.apply(_style_status, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download template CSV",
            data=",".join([r[0] for r in SCHEMA_TABLE]) + "\n",
            file_name="gl_template.csv",
            mime="text/csv",
        )

    if not uploaded_file:
        return {}

    # ── Load file ──────────────────────────────────────────────────────
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith((".xlsx", ".xls", ".xlsm")):
            # Let user pick the sheet if multiple
            xf = pd.ExcelFile(uploaded_file)
            if len(xf.sheet_names) > 1:
                sheet = st.selectbox("Select Excel sheet", xf.sheet_names)
            else:
                sheet = xf.sheet_names[0]
            raw_df = pd.read_excel(uploaded_file, sheet_name=sheet)
        else:
            enc = st.selectbox("File encoding", ["utf-8", "latin-1", "cp1252"],
                               help="Try latin-1 or cp1252 if you see garbled characters.")
            sep = st.selectbox("Delimiter", [",", ";", "\t", "|"])
            raw_df = pd.read_csv(uploaded_file, encoding=enc, sep=sep, low_memory=False)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return {}

    st.success(f"File loaded: **{len(raw_df):,} rows × {len(raw_df.columns)} columns**")
    st.dataframe(raw_df.head(5), use_container_width=True)

    # ── Auto-detect mapping ────────────────────────────────────────────
    auto = _auto_map(list(raw_df.columns))
    file_col_options = ["[ not mapped ]"] + list(raw_df.columns)

    st.markdown("#### Column Mapping")
    st.caption(
        "Auto-detected from your column headers. "
        "Adjust any mismatches. Required fields are marked."
    )

    final_mapping = {}
    all_std = _REQUIRED_COLS + _RECOMMENDED_COLS + _OPTIONAL_COLS
    cols_per_row = 3
    rows = [all_std[i:i+cols_per_row] for i in range(0, len(all_std), cols_per_row)]

    for row_cols in rows:
        ui_cols = st.columns(len(row_cols))
        for ui_col, std_col in zip(ui_cols, row_cols):
            detected = auto.get(std_col, "[ not mapped ]")
            status   = "[R]" if std_col in _REQUIRED_COLS else (
                       "[+]" if std_col in _RECOMMENDED_COLS else "[ ]")
            default_idx = (
                file_col_options.index(detected) if detected in file_col_options else 0
            )
            chosen = ui_col.selectbox(
                f"{status} {std_col}",
                options=file_col_options,
                index=default_idx,
                key=f"colmap_{std_col}",
            )
            if chosen != "[ not mapped ]":
                final_mapping[std_col] = chosen

    # Validate required columns
    missing_required = [c for c in _REQUIRED_COLS if c not in final_mapping]
    if missing_required:
        st.error(f"Required columns not mapped: **{', '.join(missing_required)}**. "
                 "Please map them above before proceeding.")
        return {}

    # Store raw_df in session state for load_data to access
    st.session_state["_uploaded_raw_df"]  = raw_df
    st.session_state["_col_mapping"]       = final_mapping
    st.success(f"Mapping confirmed: {len(final_mapping)} columns mapped.")
    return final_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & filtering
# ─────────────────────────────────────────────────────────────────────────────
def load_data(cfg: dict) -> pd.DataFrame:
    if cfg["data_source"] == "Upload CSV / Excel":
        raw_df  = st.session_state.get("_uploaded_raw_df")
        mapping = st.session_state.get("_col_mapping", {})
        if raw_df is None or not mapping:
            return None
        # Apply column mapping
        rename = {v: k for k, v in mapping.items() if v != k}
        df = raw_df.rename(columns=rename)
        return _parse_df(df)
    return load_sample_data()


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    # Date range
    if len(cfg["date_range"]) == 2:
        start, end = cfg["date_range"]
        out = out[
            (out["posting_date"].dt.date >= start) &
            (out["posting_date"].dt.date <= end)
        ]

    # Entity
    if cfg["filter_entity"] and "entity" in out.columns:
        out = out[out["entity"].isin(cfg["filter_entity"])]

    # Account type
    if cfg["filter_acct_type"] and "account_type" in out.columns:
        out = out[out["account_type"].isin(cfg["filter_acct_type"])]

    # User
    if cfg["filter_user"] and "user_id" in out.columns:
        out = out[out["user_id"].isin(cfg["filter_user"])]

    # Debit / Credit
    if cfg["filter_dc"] != "Both" and "debit_credit" in out.columns:
        code = "D" if "Debit" in cfg["filter_dc"] else "C"
        out = out[out["debit_credit"] == code]

    # Amount range
    if cfg["amount_min"] > 0:
        out = out[out["amount"].abs() >= cfg["amount_min"]]
    if cfg["amount_max"] > 0:
        out = out[out["amount"].abs() <= cfg["amount_max"]]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Executive Summary
# ─────────────────────────────────────────────────────────────────────────────
def tab_executive(df: pd.DataFrame, scored_df: pd.DataFrame,
                  benford_result, ml_scored: pd.DataFrame, cfg: dict):
    st.markdown('<div class="section-header">Executive Risk Summary</div>', unsafe_allow_html=True)

    total       = len(df)
    flagged_je  = int((scored_df["risk_score"] >= 40).sum()) if scored_df is not None else 0
    flagged_pct = flagged_je / total * 100 if total else 0
    ml_anom     = int(ml_scored["is_anomaly"].sum()) if ml_scored is not None else 0
    total_amt   = df["amount"].abs().sum()
    flagged_amt = (
        scored_df.loc[scored_df["risk_score"] >= 40, "amount"].abs().sum()
        if scored_df is not None else 0
    )

    # Compute overall risk
    b_risk = benford_result.isa_240_risk_level if benford_result else "N/A"
    if b_risk == "HIGH" or flagged_pct > 10:
        overall = "HIGH"
        badge = "badge-high"
    elif b_risk == "MEDIUM" or flagged_pct > 3:
        overall = "MEDIUM"
        badge = "badge-medium"
    else:
        overall = "LOW"
        badge = "badge-low"

    st.markdown(
        f'<div class="metric-card"><div class="label">Overall Engagement Risk</div>'
        f'<div class="value"><span class="{badge}">&nbsp;{overall}&nbsp;</span></div>'
        f'<div class="delta">Benford Risk: {b_risk}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total JE Analysed",   f"{total:,}")
        st.metric("Total Amount",        f"${total_amt:,.0f}")
    with col2:
        st.metric("Flagged Entries",     f"{flagged_je:,}",  delta=f"{flagged_pct:.1f}% of total")
        st.metric("Flagged Amount",      f"${flagged_amt:,.0f}")
    with col3:
        st.metric("ML Anomalies",        f"{ml_anom:,}",
                  delta=f"{ml_anom/total*100:.1f}% rate" if total else "")
        if benford_result:
            st.metric("Benford MAD",     f"{benford_result.mad:.5f}",
                      delta=benford_result.mad_conformity)
    with col4:
        if benford_result:
            st.metric("Chi² p-value",    f"{benford_result.chi_square_p:.4f}")
        if scored_df is not None:
            crit = int((scored_df["risk_level"] == "Critical").sum())
            st.metric("Critical Risk JE", f"{crit:,}")

    # Risk level distribution
    if scored_df is not None and "risk_level" in scored_df.columns:
        st.markdown('<div class="section-header">Risk Level Distribution</div>', unsafe_allow_html=True)
        rl = scored_df["risk_level"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        ).reset_index()
        rl.columns = ["Risk Level", "Count"]
        color_map = {"Critical": "#DC2626", "High": "#EA580C", "Medium": "#D97706", "Low": "#059669"}
        fig = px.bar(
            rl, x="Risk Level", y="Count",
            color="Risk Level", color_discrete_map=color_map,
            text="Count",
        )
        fig.update_layout(
            height=320, showlegend=False,
            plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
            margin=dict(t=10, b=10),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Top flags
    if scored_df is not None:
        st.markdown('<div class="section-header">Top Risk Indicators</div>', unsafe_allow_html=True)
        flag_cols = [c for c in scored_df.columns if c.startswith("flag_")]
        flag_summary = pd.DataFrame({
            "Indicator": [c.replace("flag_", "").replace("_", " ").title() for c in flag_cols],
            "Count": [int(scored_df[c].sum()) for c in flag_cols],
            "Amount ($)": [
                f"${scored_df.loc[scored_df[c], 'amount'].abs().sum():,.0f}"
                for c in flag_cols
            ],
        }).sort_values("Count", ascending=False).head(8)
        st.dataframe(flag_summary, use_container_width=True, hide_index=True)

    # Timeline
    st.markdown('<div class="section-header">Flagged Entries Over Time</div>', unsafe_allow_html=True)
    if scored_df is not None:
        ts = scored_df[scored_df["risk_score"] >= 40].copy()
        ts["month"] = ts["posting_date"].dt.to_period("M").astype(str)
        ts_grp = ts.groupby("month").agg(count=("risk_score", "size"), avg_risk=("risk_score", "mean"))
        fig2 = px.bar(
            ts_grp.reset_index(), x="month", y="count",
            color="avg_risk",
            color_continuous_scale=["#059669", "#D97706", "#DC2626"],
            labels={"count": "Flagged Entries", "month": "Period", "avg_risk": "Avg Risk Score"},
        )
        fig2.update_layout(height=300, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Benford's Law
# ─────────────────────────────────────────────────────────────────────────────
def tab_benford(df: pd.DataFrame, benford_result, cfg: dict):
    st.markdown('<div class="section-header">Benford\'s Law Conformity Analysis</div>',
                unsafe_allow_html=True)

    if benford_result is None:
        st.warning("Benford analysis could not be computed: insufficient data.")
        return

    r = benford_result

    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations (n)", f"{r.n:,}")
        st.metric("Chi² statistic", f"{r.chi_square:.2f}")
    with col2:
        st.metric("Chi² p-value", f"{r.chi_square_p:.4f}",
                  delta="Significant" if r.chi_square_p < cfg["alpha"] else "Not significant")
        st.metric("Critical value (95%)", f"{r.chi_square_critical:.2f}")
    with col3:
        st.metric("MAD", f"{r.mad:.5f}")
        st.metric("Conformity", r.mad_conformity)
    with col4:
        st.metric("KS statistic", f"{r.ks_statistic:.4f}")
        badge = r.isa_240_risk_level
        st.metric("ISA 240 Risk", badge)

    # ISA commentary
    st.markdown(
        f'<div class="finding-box"><strong>ISA 240 Commentary:</strong> {r.isa_240_comment}</div>',
        unsafe_allow_html=True,
    )

    # Main Benford chart
    digits = sorted(r.expected.keys())
    exp_pct = [r.expected[d] * 100 for d in digits]
    obs_pct = [r.observed.get(d, 0) * 100 for d in digits]
    z_vals  = [r.z_scores.get(d, 0) for d in digits]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Expected vs Observed Distribution",
        "Z-Score per Digit (|Z| > 1.96 = significant)"
    ))
    fig.add_trace(go.Bar(name="Expected (Benford)", x=[str(d) for d in digits],
                          y=exp_pct, marker_color="#CBD5E1",
                          hovertemplate="%{y:.3f}%"), row=1, col=1)
    bar_colors = ["#DC2626" if d in r.red_flag_digits else "#FFE600" for d in digits]
    fig.add_trace(go.Bar(name="Observed", x=[str(d) for d in digits],
                          y=obs_pct, marker_color=bar_colors,
                          hovertemplate="%{y:.3f}%"), row=1, col=1)

    z_colors = ["#DC2626" if z > 1.96 else "#3B82F6" for z in z_vals]
    fig.add_trace(go.Bar(name="Z-Score", x=[str(d) for d in digits],
                          y=z_vals, marker_color=z_colors,
                          showlegend=False), row=1, col=2)
    fig.add_hline(y=1.96, line_dash="dash", line_color="#EA580C",
                  annotation_text="Z=1.96 (5% sig.)", row=1, col=2)

    fig.update_layout(
        height=420, barmode="group", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Red flags
    if r.red_flag_digits:
        st.markdown(
            f'<div class="finding-box"><strong>Red Flag Digits:</strong> '
            f'{", ".join(str(d) for d in r.red_flag_digits)}. '
            f'Z-score > 1.96. Investigate entries with these leading digits.</div>',
            unsafe_allow_html=True,
        )

    # Segment analysis
    if cfg["segment_by"] != "None" and cfg["segment_by"] in df.columns:
        st.markdown('<div class="section-header">Segment Analysis</div>', unsafe_allow_html=True)
        analyzer = BenfordAnalyzer()
        seg_df = analyzer.segment_analysis(
            df, "amount", cfg["segment_by"],
            digit_type=cfg["digit_type"],
            alpha=cfg["alpha"],
            min_amount=cfg["benford_min"],
        )
        if not seg_df.empty:
            def color_risk(row):
                c = {"HIGH": "background-color:#FEE2E2",
                     "MEDIUM": "background-color:#FEF3C7",
                     "LOW": "background-color:#D1FAE5"}.get(row["ISA 240 Risk"], "")
                return [c] * len(row)
            styled = seg_df.style.apply(color_risk, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # Deviation table
    st.markdown('<div class="section-header">Digit-Level Detail</div>', unsafe_allow_html=True)
    detail = pd.DataFrame({
        "Digit":       [str(d) for d in digits],
        "Expected %":  [f"{r.expected[d]*100:.3f}%" for d in digits],
        "Observed %":  [f"{r.observed.get(d,0)*100:.3f}%" for d in digits],
        "Deviation":   [f"{(r.observed.get(d,0)-r.expected[d])*100:+.3f}%" for d in digits],
        "Z-Score":     [f"{r.z_scores.get(d,0):.3f}" for d in digits],
        "Flag":        ["RED FLAG" if d in r.red_flag_digits else "" for d in digits],
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Journal Entry Risk
# ─────────────────────────────────────────────────────────────────────────────
def tab_je_risk(scored_df: pd.DataFrame, cfg: dict):
    st.markdown('<div class="section-header">Journal Entry Risk Scoring</div>',
                unsafe_allow_html=True)

    if scored_df is None or scored_df.empty:
        st.warning("No data to display.")
        return

    # Threshold picker
    score_threshold = st.slider(
        "Display entries with risk score ≥", min_value=0, max_value=100, value=30, step=5
    )
    display_df = scored_df[scored_df["risk_score"] >= score_threshold].sort_values(
        "risk_score", ascending=False
    )

    st.metric("Entries above threshold", f"{len(display_df):,}",
              delta=f"{len(display_df)/len(scored_df)*100:.1f}% of population")

    # Risk scatter
    fig = px.scatter(
        display_df.head(2000),
        x="posting_date",
        y="risk_score",
        color="risk_level",
        size=np.log1p(display_df.head(2000)["amount"].abs()) * 3 + 3,
        hover_data={
            c: True for c in ["je_id", "entity", "user_id", "description", "amount"]
            if c in display_df.columns
        },
        color_discrete_map={
            "Critical": "#DC2626", "High": "#EA580C",
            "Medium": "#D97706", "Low": "#059669",
        },
        labels={"risk_score": "Risk Score (0–100)", "posting_date": "Posting Date"},
        title="Journal Entry Risk Map",
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#DC2626",
                  annotation_text="Critical threshold")
    fig.add_hline(y=40, line_dash="dash", line_color="#D97706",
                  annotation_text="High threshold")
    fig.update_layout(height=400, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Flag indicator breakdown
    st.markdown('<div class="section-header">Risk Indicator Breakdown</div>',
                unsafe_allow_html=True)
    flag_cols = [c for c in scored_df.columns if c.startswith("flag_")]
    indicator_df = pd.DataFrame({
        "Indicator": [c.replace("flag_", "").replace("_", " ").title() for c in flag_cols],
        "Count":     [int(scored_df[c].sum()) for c in flag_cols],
        "Pct (%)":   [round(scored_df[c].mean() * 100, 2) for c in flag_cols],
        "Avg Amount ($)": [
            round(scored_df.loc[scored_df[c], "amount"].abs().mean(), 0)
            if scored_df[c].sum() > 0 else 0
            for c in flag_cols
        ],
    }).sort_values("Count", ascending=False)

    fig2 = px.bar(
        indicator_df, x="Count", y="Indicator", orientation="h",
        color="Count", color_continuous_scale=["#059669", "#D97706", "#DC2626"],
        title="Flagged entries per indicator",
    )
    fig2.update_layout(height=400, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                       coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # Filterable table
    st.markdown('<div class="section-header">Flagged Journal Entries</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        level_filter = st.multiselect(
            "Filter by risk level",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with col2:
        flag_filter = st.multiselect(
            "Filter by indicator",
            [c.replace("flag_", "").replace("_", " ").title() for c in flag_cols],
            default=[],
        )

    tbl = display_df.copy()
    if level_filter:
        tbl = tbl[tbl["risk_level"].isin(level_filter)]
    if flag_filter:
        selected_flags = ["flag_" + f.replace(" ", "_").lower() for f in flag_filter]
        mask = tbl[selected_flags].any(axis=1)
        tbl = tbl[mask]

    display_cols = [c for c in [
        "je_id", "entity", "posting_date", "account_code", "description",
        "debit_credit", "amount", "currency", "user_id",
        "posting_hour", "risk_score", "risk_level", "flag_count",
    ] if c in tbl.columns]

    def style_risk(row):
        color = {
            "Critical": "#FEE2E2", "High": "#FFEDD5",
            "Medium": "#FEF3C7", "Low": "#D1FAE5",
        }.get(str(row.get("risk_level", "")), "")
        return [f"background-color:{color}" if color else ""] * len(row)

    styled = tbl[display_cols].head(500).style.apply(style_risk, axis=1).format(
        {"amount": "${:,.2f}", "risk_score": "{:.0f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"Showing top 500 of {len(tbl):,} matching entries.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab: ML Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
def tab_ml(ml_scored: pd.DataFrame, scored_df: pd.DataFrame, cfg: dict):
    st.markdown('<div class="section-header">Machine Learning Anomaly Detection</div>',
                unsafe_allow_html=True)

    if ml_scored is None:
        st.warning("ML results not available.")
        return

    n_total   = len(ml_scored)
    n_anomaly = int(ml_scored["is_anomaly"].sum())
    rate      = n_anomaly / n_total * 100 if n_total else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total entries", f"{n_total:,}")
        st.metric("Anomalies detected", f"{n_anomaly:,}")
    with col2:
        st.metric("Anomaly rate", f"{rate:.2f}%")
        avg_score = ml_scored.loc[ml_scored["is_anomaly"], "anomaly_score"].mean()
        st.metric("Avg anomaly score", f"{avg_score:.3f}")
    with col3:
        # Overlap with statistical
        if scored_df is not None and "risk_score" in scored_df.columns:
            stat_flags = scored_df["risk_score"] >= 40
            detector = AnomalyDetector()
            overlap = detector.overlap_analysis(ml_scored["is_anomaly"], stat_flags)
            st.metric("Consensus (ML + Stat)", f"{overlap['consensus']:,}")
            st.metric("ML only", f"{overlap['ml_only']:,}")

    st.markdown(
        '<div class="finding-box"><strong>Isolation Forest</strong> detects multivariate outliers '
        'using features: log-amount, posting time, day-of-week, account type, user profile. '
        'SHAP values identify the primary driver for each anomaly (ISA 240.A26).</div>',
        unsafe_allow_html=True,
    )

    # Anomaly scatter
    fig = px.scatter(
        ml_scored.head(3000),
        x="posting_date",
        y="anomaly_score",
        color="is_anomaly",
        color_discrete_map={True: "#DC2626", False: "#94A3B8"},
        size=np.log1p(ml_scored.head(3000)["amount"].abs()) * 2 + 2,
        hover_data={
            c: True for c in ["je_id", "entity", "amount", "user_id",
                               "top_risk_driver", "anomaly_score"]
            if c in ml_scored.columns
        },
        labels={"anomaly_score": "Anomaly Score", "posting_date": "Date",
                "is_anomaly": "Anomaly"},
        title="Amount vs Anomaly Score Over Time",
    )
    fig.update_layout(height=400, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Amount vs score
    fig2 = px.scatter(
        ml_scored[ml_scored["is_anomaly"]].head(1000),
        x="amount",
        y="anomaly_score",
        color="top_risk_driver" if "top_risk_driver" in ml_scored.columns else None,
        hover_data={
            c: True for c in ["je_id", "entity", "user_id", "description"]
            if c in ml_scored.columns
        },
        title="Anomalous Entries: Amount vs Score",
        labels={"amount": "Amount ($)", "anomaly_score": "Anomaly Score"},
        log_x=True,
    )
    fig2.update_layout(height=380, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                       margin=dict(t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # Top anomalies table
    st.markdown('<div class="section-header">Top Anomalies (ranked by score)</div>',
                unsafe_allow_html=True)
    top_anom = ml_scored[ml_scored["is_anomaly"]].sort_values(
        "anomaly_score", ascending=False
    ).head(200)
    display_cols = [c for c in [
        "je_id", "entity", "posting_date", "account_code", "description",
        "amount", "user_id", "posting_hour",
        "anomaly_score", "anomaly_percentile", "top_risk_driver",
    ] if c in top_anom.columns]
    st.dataframe(
        top_anom[display_cols].style.format({
            "amount": "${:,.2f}",
            "anomaly_score": "{:.4f}",
            "anomaly_percentile": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # SHAP / driver distribution
    if "top_risk_driver" in ml_scored.columns and ml_scored["top_risk_driver"].ne("N/A").any():
        st.markdown('<div class="section-header">Primary Risk Drivers (SHAP)</div>',
                    unsafe_allow_html=True)
        driver_counts = (
            ml_scored[ml_scored["is_anomaly"]]["top_risk_driver"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        driver_counts.columns = ["Feature", "Count"]
        fig3 = px.bar(
            driver_counts, x="Count", y="Feature", orientation="h",
            color="Count", color_continuous_scale=["#3B82F6", "#DC2626"],
            title="Most Common SHAP Risk Driver Among Anomalies",
        )
        fig3.update_layout(height=350, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                           coloraxis_showscale=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Vendor Analysis
# ─────────────────────────────────────────────────────────────────────────────
def tab_vendors(df: pd.DataFrame, cfg: dict):
    st.markdown('<div class="section-header">Vendor / Counterparty Risk Analysis</div>',
                unsafe_allow_html=True)

    analyzer = VendorAnalyzer(
        duplicate_window_days=cfg["dup_window"],
        fuzzy_threshold=cfg["fuzzy_threshold"],
    )

    # ── Duplicates ─────────────────────────────────────────────────────
    st.markdown("##### Duplicate Payment Detection")
    with st.spinner("Detecting duplicate payments..."):
        dups = analyzer.detect_duplicates(df)
    if not dups.empty:
        st.error(f"{len(dups):,} potential duplicate payments detected "
                 f"(total: ${dups['amount'].abs().sum():,.0f})")
        st.dataframe(
            dups[[c for c in ["je_id", "entity", "posting_date", "vendor_name",
                               "amount", "invoice_ref", "duplicate_group"]
                   if c in dups.columns]].head(200),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No duplicate payments detected within the selected window.")

    # ── Ghost vendors ──────────────────────────────────────────────────
    st.markdown("##### Ghost Vendor Detection (Fuzzy Name Matching)")
    with st.spinner("Running fuzzy name matching..."):
        ghosts = analyzer.detect_ghost_vendors(df)
    if not ghosts.empty:
        st.warning(f"{len(ghosts)} suspicious vendor name pairs found (similarity >= {cfg['fuzzy_threshold']}%)")
        st.dataframe(ghosts, use_container_width=True, hide_index=True)
    else:
        st.success("No ghost vendors detected at current similarity threshold.")

    # ── Concentration ──────────────────────────────────────────────────
    st.markdown("##### Vendor Concentration Analysis")
    concentration = analyzer.concentration_analysis(df)
    if not concentration.empty:
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(
                concentration.head(10),
                x="vendor_name",
                y="pct_of_total",
                text="pct_of_total",
                color="pct_of_total",
                color_continuous_scale=["#059669", "#D97706", "#DC2626"],
                title="Top 10 Vendors: % of Total Spend",
                labels={"pct_of_total": "% of Total Spend", "vendor_name": "Vendor"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=380, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                              coloraxis_showscale=False, margin=dict(t=40, b=10),
                              xaxis_tickangle=30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(
                concentration.style.format({
                    "total_amount": "${:,.0f}",
                    "avg_transaction": "${:,.0f}",
                    "pct_of_total": "{:.1f}%",
                    "cumulative_pct": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ── Timing ─────────────────────────────────────────────────────────
    st.markdown("##### Payment Timing Analysis")
    by_day, by_hour = analyzer.payment_timing(df)
    col1, col2 = st.columns(2)
    with col1:
        if not by_day.empty:
            fig_d = px.bar(
                by_day, x="day_name", y="count",
                color="count", color_continuous_scale=["#3B82F6", "#DC2626"],
                title="Transactions by Day of Week",
            )
            fig_d.update_layout(height=320, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                                coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_d, use_container_width=True)
    with col2:
        if not by_hour.empty:
            fig_h = px.bar(
                by_hour, x="time_window", y="count",
                color="count", color_continuous_scale=["#059669", "#DC2626"],
                title="Transactions by Time Window",
            )
            fig_h.update_layout(height=320, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
                                coloraxis_showscale=False, margin=dict(t=40, b=10),
                                xaxis_tickangle=25)
            st.plotly_chart(fig_h, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Audit Report
# ─────────────────────────────────────────────────────────────────────────────
def _build_findings(df, scored_df, benford_result, ml_scored, cfg):
    """Compile all analysis results into a single dict for reporting."""
    total       = len(df)
    flagged     = int((scored_df["risk_score"] >= 40).sum()) if scored_df is not None else 0
    critical_je = int((scored_df["risk_score"] >= 70).sum()) if scored_df is not None else 0
    flagged_pct = flagged / total * 100 if total else 0

    je_stats = {
        "total_entries":  total,
        "flagged_entries": flagged,
        "critical_je":    critical_je,
        "top_indicators": {},
    }
    if scored_df is not None:
        flag_cols = [c for c in scored_df.columns if c.startswith("flag_")]
        top5 = sorted(flag_cols, key=lambda c: scored_df[c].sum(), reverse=True)[:5]
        for c in top5:
            label = c.replace("flag_", "").replace("_", " ").title()
            je_stats["top_indicators"][label] = f"{int(scored_df[c].sum()):,} entries"

    ml_stats = {
        "anomaly_count": int(ml_scored["is_anomaly"].sum()) if ml_scored is not None else 0,
        "consensus":     0,
        "top_drivers":   "N/A",
    }
    if ml_scored is not None and scored_df is not None:
        ml_stats["consensus"] = int(
            (ml_scored["is_anomaly"] & (scored_df["risk_score"] >= 40)).sum()
        )
    if ml_scored is not None and "top_risk_driver" in ml_scored.columns:
        top_drv = (
            ml_scored[ml_scored["is_anomaly"]]["top_risk_driver"]
            .value_counts().head(3).index.tolist()
        )
        ml_stats["top_drivers"] = ", ".join(top_drv) or "N/A"

    analyzer = VendorAnalyzer(duplicate_window_days=cfg["dup_window"])
    dups   = analyzer.detect_duplicates(df)
    ghosts = analyzer.detect_ghost_vendors(df)
    conc   = analyzer.concentration_analysis(df)
    vendor_stats = {
        "duplicate_count":   len(dups),
        "ghost_vendor_pairs": len(ghosts),
        "top_vendor":        conc.iloc[0]["vendor_name"] if not conc.empty else "N/A",
        "top_vendor_pct":    float(conc.iloc[0]["pct_of_total"]) if not conc.empty else 0.0,
    }

    b_risk    = benford_result.isa_240_risk_level if benford_result else "N/A"
    b_comment = benford_result.isa_240_comment    if benford_result else ""

    # Auto-ranked findings
    top_findings = []
    if b_risk == "HIGH":
        top_findings.append({
            "risk_level": "High",
            "title":       "Benford's Law Non-Conformity",
            "description": "Statistically significant deviation from Benford's Law. Possible fabricated or manipulated amounts.",
            "isa_ref":     "ISA 240.A4, ISA 500.A14",
            "action":      "Expand substantive testing on red-flag digit populations.",
        })
    if flagged_pct > 5:
        top_findings.append({
            "risk_level": "High",
            "title":       f"Elevated JE Risk Indicators ({flagged_pct:.1f}% flagged)",
            "description": f"{flagged:,} entries exceeded risk threshold: round numbers, period-end, off-hours.",
            "isa_ref":     "ISA 240.A3",
            "action":      "Targeted testing on High/Critical entries; focus on period-end.",
        })
    if len(dups) > 0:
        top_findings.append({
            "risk_level": "High",
            "title":       f"Potential Duplicate Payments ({len(dups):,} entries)",
            "description": f"Same vendor + amount within {cfg['dup_window']} days.",
            "isa_ref":     "ISA 240.A3, COSO Fraud Framework",
            "action":      "Inspect underlying invoices and payment confirmations.",
        })
    if len(ghosts) > 0:
        top_findings.append({
            "risk_level": "Medium",
            "title":       f"Ghost Vendor Risk ({len(ghosts)} pairs)",
            "description": "High-similarity vendor names detected. Potential ghost vendor scheme.",
            "isa_ref":     "ISA 240.A3",
            "action":      "Verify vendor master data, bank details, approval chain.",
        })
    if ml_stats["anomaly_count"] > 0:
        top_findings.append({
            "risk_level": "Medium",
            "title":       f"ML Anomalies ({ml_stats['anomaly_count']:,} entries)",
            "description": f"Isolation Forest outliers; {ml_stats['consensus']:,} consensus with statistical flags.",
            "isa_ref":     "ISA 240.A26, PCAOB AS 2401.66",
            "action":      "Review top-scoring anomalies, prioritise consensus entries.",
        })
    if not top_findings:
        top_findings.append({
            "risk_level": "Low",
            "title":       "No significant findings",
            "description": "Data analytics procedures did not identify material risk indicators.",
            "isa_ref":     "ISA 500.A1",
            "action":      "Continue with standard audit procedures.",
        })

    return {
        "cfg":          cfg,
        "digit_type":   cfg.get("digit_type", "first"),
        "benford":      benford_result,
        "je_stats":     je_stats,
        "ml_stats":     ml_stats,
        "vendor_stats": vendor_stats,
        "top_findings": top_findings,
        "_dups":        dups,
        "_ghosts":      ghosts,
        "_flagged_pct": flagged_pct,
        "_b_risk":      b_risk,
        "_b_comment":   b_comment,
    }


def tab_report(
    df: pd.DataFrame,
    scored_df: pd.DataFrame,
    benford_result,
    ml_scored: pd.DataFrame,
    cfg: dict,
):
    st.markdown('<div class="section-header">Audit Risk Memorandum</div>',
                unsafe_allow_html=True)

    findings = _build_findings(df, scored_df, benford_result, ml_scored, cfg)
    dups      = findings["_dups"]
    ghosts    = findings["_ghosts"]
    b_risk    = findings["_b_risk"]
    b_comment = findings["_b_comment"]
    top_findings = findings["top_findings"]

    reporter = RiskReporter(
        client_name=cfg["client_name"],
        engagement_partner=cfg["partner"],
        audit_period=cfg["audit_period"],
        materiality=cfg["materiality"],
        prepared_by="Data Analytics Team",
    )

    # ── Static memo (always available) ────────────────────────────────
    memo_text = reporter.generate_text_memo(
        b_risk, b_comment,
        findings["je_stats"], findings["ml_stats"], findings["vendor_stats"],
        top_findings,
    )

    # ── AI generation section ──────────────────────────────────────────
    st.markdown('<div class="section-header">AI-Powered Memo Generation</div>',
                unsafe_allow_html=True)

    ai_col1, ai_col2 = st.columns([3, 1])
    with ai_col1:
        provider_label = PROVIDERS.get(cfg["ai_provider"], {}).get("label", cfg["ai_provider"])
        st.markdown(
            f'<div class="finding-box"><strong>Provider:</strong> {provider_label} &nbsp;|&nbsp; '
            f'<strong>Model:</strong> {cfg["ai_model"]}<br/>'
            f'Generate a professional ISA-referenced audit memorandum using AI. '
            f'The model receives all analysis results and produces a Big 4 quality memo.</div>',
            unsafe_allow_html=True,
        )
    with ai_col2:
        generate_ai = st.button("Generate AI Memo", type="primary", use_container_width=True)

    ai_memo_key = "ai_memo_text"

    if generate_ai:
        if not cfg.get("ai_api_key", "").strip():
            st.error("Please enter your API key in the sidebar before generating.")
        else:
            with st.spinner(f"Generating memo with {cfg['ai_model']}..."):
                try:
                    ai_reporter = AIReporter()
                    ai_text = ai_reporter.generate(
                        findings=findings,
                        provider=cfg["ai_provider"],
                        model=cfg["ai_model"],
                        api_key=cfg["ai_api_key"],
                    )
                    st.session_state[ai_memo_key] = ai_text
                    st.success("AI memo generated successfully.")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if ai_memo_key in st.session_state and st.session_state[ai_memo_key]:
        st.markdown("##### AI-Generated Audit Memorandum")
        st.markdown(st.session_state[ai_memo_key])
        st.download_button(
            "Download AI Memo (Markdown)",
            data=st.session_state[ai_memo_key],
            file_name=f"AI_AuditMemo_{cfg['audit_period'].replace(' ','_')}.md",
            mime="text/markdown",
        )
        st.divider()

    # ── Static memo fallback ───────────────────────────────────────────
    st.markdown("##### Standard Audit Memorandum (DA-001)")
    st.text_area("", memo_text, height=500, label_visibility="collapsed")

    # ── Ranked findings display ────────────────────────────────────────
    st.markdown('<div class="section-header">Ranked Findings</div>', unsafe_allow_html=True)
    for i, f in enumerate(top_findings, 1):
        badge_map = {"High": "badge-high", "Critical": "badge-critical",
                     "Medium": "badge-medium", "Low": "badge-low"}
        badge = badge_map.get(f["risk_level"], "badge-low")
        st.markdown(
            f'<div class="finding-box">'
            f'<strong>[{i}] <span class="{badge}">&nbsp;{f["risk_level"].upper()}&nbsp;</span>'
            f': {f["title"]}</strong><br/>'
            f'{f["description"]}<br/>'
            f'<em>ISA Ref:</em> {f["isa_ref"]} &nbsp;|&nbsp; '
            f'<em>Action:</em> {f["action"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Export buttons
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        # Excel export
        try:
            excel_bytes = reporter.export_excel(
                je_scored=scored_df,
                benford_result=benford_result,
                ml_scored=ml_scored,
                vendor_duplicates=dups if not dups.empty else None,
                ghost_vendors=ghosts if not ghosts.empty else None,
            )
            st.download_button(
                label="Download Excel Workbook",
                data=excel_bytes,
                file_name=f"AuditAI_{cfg['client_name'].replace(' ','_')}_Exceptions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")

    with col2:
        # Text memo download
        st.download_button(
            label="Download Audit Memo (TXT)",
            data=memo_text.encode("utf-8"),
            file_name=f"AuditMemo_{cfg['audit_period']}.txt",
            mime="text/plain",
        )

    with col3:
        # CSV of flagged entries
        if scored_df is not None:
            flagged_csv = scored_df[scored_df["risk_score"] >= 40].to_csv(index=False)
            st.download_button(
                label="Download Flagged JE (CSV)",
                data=flagged_csv,
                file_name=f"Flagged_JE_{cfg['audit_period']}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = build_sidebar()

    # Header
    st.markdown(
        f'<div class="ey-banner">'
        f'<h1>AuditAI Analytics</h1>'
        f'<p>Audit Data Analytics Platform &nbsp;|&nbsp; '
        f'ISA 240  ·  ISA 315  ·  ISA 500  ·  PCAOB AS 2401 &nbsp;|&nbsp; '
        f'Client: <strong>{cfg["client_name"]}</strong> &nbsp;|&nbsp; '
        f'Period: <strong>{cfg["audit_period"]}</strong></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Upload guide + column mapping (only when upload mode) ──────────
    if cfg["data_source"] == "Upload CSV / Excel":
        render_data_guide_and_mapping(cfg["uploaded_file"])
        # Gate: don't proceed until mapping is confirmed
        if not st.session_state.get("_col_mapping"):
            st.info("Upload your GL file and confirm the column mapping above to start the analysis.")
            return

    # Load data
    with st.spinner("Loading data..."):
        raw_df = load_data(cfg)

    if raw_df is None or raw_df.empty:
        if cfg["data_source"] == "Upload CSV / Excel":
            st.warning("File not yet loaded or column mapping incomplete.")
        else:
            st.warning("No data loaded. Please use the sample dataset or upload a file.")
        return

    # Store for filter option population on next rerun
    st.session_state["_raw_df"] = raw_df

    # Apply filters
    df = apply_filters(raw_df, cfg)

    if df.empty:
        st.error("No entries match the current filters. Please adjust the criteria.")
        return

    st.info(
        f"**Dataset:** {len(df):,} entries "
        f"({'sample data' if cfg['data_source'].startswith('Sample') else 'uploaded file'}) "
        f"| Filtered from {len(raw_df):,} total."
    )

    # Run analyses (cached)
    benford_result = None
    try:
        benford_result = run_benford(df, cfg["digit_type"], cfg["benford_min"], cfg["alpha"])
    except Exception as e:
        st.sidebar.warning(f"Benford: {e}")

    scored_df = None
    try:
        scored_df = run_je_scoring(
            df, cfg["materiality"], cfg["period_end_days"],
            cfg["unusual_start"], cfg["unusual_end"], cfg["holiday_country"]
        )
    except Exception as e:
        st.sidebar.warning(f"JE scoring: {e}")

    ml_scored = None
    try:
        ml_scored = run_ml_detection(df, cfg["contamination"], cfg["n_estimators"])
    except Exception as e:
        st.sidebar.warning(f"ML detection: {e}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Executive Summary",
        "Benford's Law",
        "Journal Entry Risk",
        "ML Anomaly Detection",
        "Vendor Analysis",
        "Audit Report",
        "Assistant",
    ])

    with tab1:
        tab_executive(df, scored_df, benford_result, ml_scored, cfg)

    with tab2:
        tab_benford(df, benford_result, cfg)

    with tab3:
        if scored_df is not None:
            tab_je_risk(scored_df, cfg)
        else:
            st.warning("Journal entry scoring unavailable.")

    with tab4:
        if ml_scored is not None:
            tab_ml(ml_scored, scored_df, cfg)
        else:
            st.warning("ML results unavailable.")

    with tab5:
        tab_vendors(df, cfg)

    with tab6:
        tab_report(df, scored_df, benford_result, ml_scored, cfg)

    with tab7:
        tab_assistant(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Assistant
# ─────────────────────────────────────────────────────────────────────────────
def tab_assistant(cfg: dict):
    st.markdown('<div class="section-header">Audit Concepts Assistant</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Ask any question about audit methodology, ISA standards, statistical tests, "
        "machine learning concepts, or how to interpret the platform results. "
        "The assistant responds in the language of your question."
    )

    api_key = cfg.get("ai_api_key", "").strip()
    if not api_key:
        st.warning(
            "Enter your API key in the sidebar (AI Report Generation section) "
            "to activate the assistant."
        )
        st.markdown("**Sample questions you can ask once connected:**")
        for q in STARTER_QUESTIONS:
            st.markdown(f"- {q}")
        return

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Starter question chips
    if not st.session_state["chat_history"]:
        st.markdown("**Suggested questions:**")
        cols = st.columns(2)
        for i, q in enumerate(STARTER_QUESTIONS[:6]):
            if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    try:
                        answer = call_chatbot(
                            history=[],
                            user_message=q,
                            provider=cfg["ai_provider"],
                            model=cfg["ai_model"],
                            api_key=api_key,
                        )
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()

    # Display history
    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Input
    user_input = st.chat_input("Ask a question about audit concepts or this analysis...")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = call_chatbot(
                        history=st.session_state["chat_history"][:-1],
                        user_message=user_input,
                        provider=cfg["ai_provider"],
                        model=cfg["ai_model"],
                        api_key=api_key,
                    )
                    st.markdown(answer)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    # Clear button
    if st.session_state["chat_history"]:
        if st.button("Clear conversation"):
            st.session_state["chat_history"] = []
            st.rerun()


if __name__ == "__main__":
    main()
