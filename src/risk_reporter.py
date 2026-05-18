"""
Audit risk report generator.

Produces a structured audit memo referencing ISA 240, ISA 315, ISA 500,
formatted as a professional Big 4-style internal working paper.
"""
from __future__ import annotations
import io
import textwrap
from datetime import date
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


_RISK_COLORS = {
    "Critical": "#DC2626",
    "High":     "#EA580C",
    "Medium":   "#D97706",
    "Low":      "#059669",
}

_ISA_LIBRARY = {
    "ISA 240":
        "The Auditor's Responsibilities Relating to Fraud in an Audit of Financial Statements",
    "ISA 315":
        "Identifying and Assessing the Risks of Material Misstatement through Understanding "
        "the Entity and Its Environment",
    "ISA 500":
        "Audit Evidence",
    "ISA 530":
        "Audit Sampling",
    "ISA 560":
        "Subsequent Events",
    "PCAOB AS 2401":
        "Consideration of Fraud in a Financial Statement Audit",
}


class RiskReporter:

    def __init__(
        self,
        client_name: str = "Client",
        engagement_partner: str = "Partner",
        audit_period: str = "",
        materiality: float = 10_000.0,
        prepared_by: str = "Data Analytics Team",
        report_date: Optional[date] = None,
    ):
        self.client_name      = client_name
        self.engagement_partner = engagement_partner
        self.audit_period     = audit_period or str(date.today().year)
        self.materiality      = materiality
        self.prepared_by      = prepared_by
        self.report_date      = report_date or date.today()

    # ------------------------------------------------------------------
    # Text memo
    # ------------------------------------------------------------------
    def generate_text_memo(
        self,
        benford_risk: str,
        benford_comment: str,
        je_stats: Dict[str, Any],
        ml_stats: Dict[str, Any],
        vendor_stats: Dict[str, Any],
        top_findings: List[Dict],
    ) -> str:
        lines = []
        sep = "=" * 72

        lines += [
            sep,
            "AUDIT DATA ANALYTICS — RISK ASSESSMENT MEMORANDUM",
            "Working Paper Reference: DA-001",
            sep,
            f"Client:            {self.client_name}",
            f"Engagement Partner:{self.engagement_partner}",
            f"Audit Period:      {self.audit_period}",
            f"Materiality:       {self.materiality:,.0f}",
            f"Prepared by:       {self.prepared_by}",
            f"Date:              {self.report_date.strftime('%d %B %Y')}",
            sep,
            "",
            "PURPOSE",
            "-" * 40,
            textwrap.fill(
                "This memorandum documents the results of data analytics procedures "
                "performed on the general ledger, in accordance with ISA 240, ISA 315, "
                "and ISA 500. The procedures include Benford's Law conformity testing, "
                "journal entry risk scoring, and machine-learning anomaly detection.",
                width=72,
            ),
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
        ]

        total_entries   = je_stats.get("total_entries", 0)
        flagged_entries = je_stats.get("flagged_entries", 0)
        flagged_pct     = (flagged_entries / total_entries * 100) if total_entries else 0
        ml_anomalies    = ml_stats.get("anomaly_count", 0)
        duplicate_count = vendor_stats.get("duplicate_count", 0)
        ghost_count     = vendor_stats.get("ghost_vendor_pairs", 0)

        overall_risk = "HIGH" if benford_risk == "HIGH" or flagged_pct > 10 else (
            "MEDIUM" if benford_risk == "MEDIUM" or flagged_pct > 3 else "LOW"
        )

        lines += [
            f"Overall Engagement Risk Level:  *** {overall_risk} ***",
            "",
            f"  Total Journal Entries Analysed:  {total_entries:,}",
            f"  Entries Flagged (JE Risk Score):  {flagged_entries:,} ({flagged_pct:.1f}%)",
            f"  ML Anomalies Detected:            {ml_anomalies:,}",
            f"  Potential Duplicate Payments:     {duplicate_count:,}",
            f"  Ghost Vendor Pairs Identified:    {ghost_count:,}",
            f"  Benford's Law Conformity Risk:    {benford_risk}",
            "",
        ]

        lines += [
            "SECTION 1 — BENFORD'S LAW ANALYSIS",
            "-" * 40,
            textwrap.fill(
                f"Benford's Law conformity testing was applied to the full population of "
                f"{total_entries:,} journal entries. The analysis produced a risk assessment "
                f"of {benford_risk}.",
                width=72,
            ),
            "",
            f"  Auditor Comment: {benford_comment}",
            "",
            "  References: ISA 240.A4, ISA 500.A14, Nigrini (2012)",
            "",
        ]

        lines += [
            "SECTION 2 — JOURNAL ENTRY TESTING",
            "-" * 40,
            textwrap.fill(
                "Automated risk indicators were applied to each journal entry in accordance "
                "with ISA 240.A3. Entries scoring above 40 are classified as High or Critical "
                "risk and require follow-up audit procedures.",
                width=72,
            ),
            "",
        ]

        for indicator, detail in je_stats.get("top_indicators", {}).items():
            lines.append(f"  {indicator:<35} {detail}")
        lines.append("")

        lines += [
            "SECTION 3 — ML ANOMALY DETECTION",
            "-" * 40,
            textwrap.fill(
                f"An Isolation Forest model was trained on {total_entries:,} journal entries "
                f"using multivariate features (amount, timing, account, user profile). "
                f"{ml_anomalies} entries were classified as anomalous.",
                width=72,
            ),
            "",
            f"  Anomaly Rate:            {ml_anomalies/total_entries*100:.2f}%" if total_entries else "",
            f"  Consensus w/ Stat Flags: {ml_stats.get('consensus', 0):,} entries",
            "",
            "  References: ISA 240.A26, PCAOB AS 2401.66",
            "",
        ]

        lines += [
            "SECTION 4 — VENDOR ANALYSIS",
            "-" * 40,
            f"  Potential Duplicate Payments:     {duplicate_count:,}",
            f"  Ghost Vendor Pairs (fuzzy match): {ghost_count:,}",
            "",
            "  References: ISA 240.A3, COSO Fraud Risk Framework",
            "",
        ]

        lines += [
            "TOP FINDINGS — RANKED BY RISK",
            "-" * 40,
        ]
        for i, finding in enumerate(top_findings, 1):
            lines += [
                f"  [{i}] {finding.get('risk_level','').upper()} — {finding.get('title','')}",
                f"      {textwrap.fill(finding.get('description',''), width=68, subsequent_indent='      ')}",
                f"      ISA Ref: {finding.get('isa_ref','')}",
                f"      Recommended Action: {finding.get('action','')}",
                "",
            ]

        lines += [
            "LIMITATIONS",
            "-" * 40,
            textwrap.fill(
                "Data analytics procedures are designed to identify anomalies and risk "
                "indicators. A flagged entry does not constitute evidence of fraud or error. "
                "All exceptions require professional judgment and further substantive "
                "procedures before audit conclusions are drawn (ISA 500.6).",
                width=72,
            ),
            "",
            "STANDARDS REFERENCED",
            "-" * 40,
        ]
        for ref, desc in _ISA_LIBRARY.items():
            lines.append(f"  {ref:<20} {desc}")
        lines += ["", sep, "END OF MEMORANDUM", sep]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------
    def export_excel(
        self,
        je_scored: pd.DataFrame,
        benford_result: Optional[Any] = None,
        ml_scored: Optional[pd.DataFrame] = None,
        vendor_duplicates: Optional[pd.DataFrame] = None,
        ghost_vendors: Optional[pd.DataFrame] = None,
    ) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            wb = writer.book

            # Formats
            fmt_header = wb.add_format({
                "bold": True, "bg_color": "#1A1A2E", "font_color": "#FFE600",
                "border": 1, "align": "center",
            })
            fmt_critical = wb.add_format({"bg_color": "#FEE2E2", "border": 1})
            fmt_high     = wb.add_format({"bg_color": "#FFEDD5", "border": 1})
            fmt_medium   = wb.add_format({"bg_color": "#FEF3C7", "border": 1})
            fmt_low      = wb.add_format({"bg_color": "#D1FAE5", "border": 1})
            fmt_title    = wb.add_format({
                "bold": True, "font_size": 14, "font_color": "#1A1A2E"
            })

            # ---- Sheet 1: JE Exceptions ----
            flagged = je_scored[je_scored["risk_score"] >= 30].copy() if je_scored is not None else pd.DataFrame()
            flagged = flagged.sort_values("risk_score", ascending=False)
            if not flagged.empty:
                flagged.to_excel(writer, sheet_name="JE Exceptions", index=False, startrow=2)
                ws = writer.sheets["JE Exceptions"]
                ws.write(0, 0, f"Journal Entry Exceptions — {self.client_name} — FY {self.audit_period}", fmt_title)
                for col_num, col_name in enumerate(flagged.columns):
                    ws.write(2, col_num, col_name, fmt_header)

            # ---- Sheet 2: Benford Summary ----
            if benford_result is not None:
                bdf = pd.DataFrame({
                    "Digit": list(benford_result.expected.keys()),
                    "Expected %": [round(v * 100, 3) for v in benford_result.expected.values()],
                    "Observed %": [round(benford_result.observed.get(d, 0) * 100, 3) for d in benford_result.expected],
                    "Deviation %": [
                        round((benford_result.observed.get(d, 0) - v) * 100, 3)
                        for d, v in benford_result.expected.items()
                    ],
                    "Z-Score": [round(benford_result.z_scores.get(d, 0), 3) for d in benford_result.expected],
                    "Red Flag": ["YES" if d in benford_result.red_flag_digits else "" for d in benford_result.expected],
                })
                bdf.to_excel(writer, sheet_name="Benford Analysis", index=False, startrow=2)
                ws2 = writer.sheets["Benford Analysis"]
                ws2.write(0, 0, f"Benford's Law Analysis — Risk: {benford_result.isa_240_risk_level}", fmt_title)
                for col_num, col_name in enumerate(bdf.columns):
                    ws2.write(2, col_num, col_name, fmt_header)

            # ---- Sheet 3: ML Anomalies ----
            if ml_scored is not None and "is_anomaly" in ml_scored.columns:
                ml_flags = ml_scored[ml_scored["is_anomaly"]].copy().sort_values(
                    "anomaly_score", ascending=False
                )
                if not ml_flags.empty:
                    display_cols = [c for c in [
                        "je_id", "entity", "posting_date", "account_code",
                        "description", "amount", "user_id",
                        "anomaly_score", "top_risk_driver"
                    ] if c in ml_flags.columns]
                    ml_flags[display_cols].to_excel(
                        writer, sheet_name="ML Anomalies", index=False, startrow=2
                    )
                    ws3 = writer.sheets["ML Anomalies"]
                    ws3.write(0, 0, "ML Anomaly Detection Results", fmt_title)
                    for col_num, col_name in enumerate(display_cols):
                        ws3.write(2, col_num, col_name, fmt_header)

            # ---- Sheet 4: Vendor Duplicates ----
            if vendor_duplicates is not None and not vendor_duplicates.empty:
                vendor_duplicates.to_excel(writer, sheet_name="Duplicate Payments", index=False, startrow=2)
                ws4 = writer.sheets["Duplicate Payments"]
                ws4.write(0, 0, "Potential Duplicate Payments", fmt_title)
                for col_num, col_name in enumerate(vendor_duplicates.columns):
                    ws4.write(2, col_num, col_name, fmt_header)

            # ---- Sheet 5: Ghost Vendors ----
            if ghost_vendors is not None and not ghost_vendors.empty:
                ghost_vendors.to_excel(writer, sheet_name="Ghost Vendors", index=False, startrow=2)
                ws5 = writer.sheets["Ghost Vendors"]
                ws5.write(0, 0, "Ghost Vendor Analysis — Fuzzy Name Matching", fmt_title)
                for col_num, col_name in enumerate(ghost_vendors.columns):
                    ws5.write(2, col_num, col_name, fmt_header)

        buffer.seek(0)
        return buffer.read()
