"""
Journal Entry risk-scoring engine.

Implements automated risk indicators aligned with:
  - ISA 240.A3 — Fraud risk factors relating to misstatements
  - PCAOB AS 2401 — Consideration of Fraud
  - Big 4 JE testing frameworks (round numbers, period-end, off-hours, etc.)
"""
import calendar
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import holidays as _hol
    _HOL_AVAILABLE = True
except ImportError:
    _HOL_AVAILABLE = False


# Risk weights (sum to max ~100 for a single entry with all flags)
_RISK_WEIGHTS: Dict[str, int] = {
    "same_day_reversal":  25,
    "round_number":       20,
    "suspense_account":   20,
    "manual_override":    15,
    "period_end":         15,
    "unusual_hour":       15,
    "unusual_user":       15,
    "weekend":            10,
    "public_holiday":     10,
    "large_amount":       10,
    "intercompany":        5,
    "split_transaction":  15,
}


class JournalEntryAnalyzer:
    """
    Scores each journal entry 0–100 across multiple red-flag indicators.

    Parameters
    ----------
    materiality : float
        Engagement materiality threshold (local currency).
    period_end_days : int
        Number of days before month/quarter/year-end to flag as period-end.
    unusual_hour_start : int
        Hour (0–23) from which postings are considered "after-hours".
    unusual_hour_end : int
        Hour (0–23) until which postings are considered "after-hours".
    country : str
        ISO 3166-1 alpha-2 country code used for public holiday detection.
    top_user_pct : float
        Minimum activity share for a user to be considered "normal".
    split_threshold : float
        Max % difference between two transactions to flag as a split payment.
    """

    def __init__(
        self,
        materiality: float = 10_000.0,
        period_end_days: int = 3,
        unusual_hour_start: int = 20,
        unusual_hour_end: int = 6,
        country: str = "US",
        top_user_pct: float = 0.01,
        split_threshold: float = 0.02,
    ):
        self.materiality = materiality
        self.period_end_days = period_end_days
        self.unusual_hour_start = unusual_hour_start
        self.unusual_hour_end = unusual_hour_end
        self.country = country
        self.top_user_pct = top_user_pct
        self.split_threshold = split_threshold

    # ------------------------------------------------------------------
    # Individual indicator helpers
    # ------------------------------------------------------------------
    def _is_round_number(self, amount: float) -> bool:
        for threshold in [100_000, 50_000, 10_000, 5_000, 1_000, 500, 100]:
            if abs(amount) >= threshold and abs(amount) % threshold == 0:
                return True
        return False

    def _is_period_end(self, dt: pd.Timestamp) -> bool:
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        days_from_end = last_day - dt.day
        if days_from_end < self.period_end_days:
            return True
        # Extra scrutiny for quarter-end and year-end
        if dt.month in (3, 6, 9, 12) and days_from_end < self.period_end_days + 2:
            return True
        return False

    def _is_unusual_hour(self, hour: int) -> bool:
        return hour >= self.unusual_hour_start or hour < self.unusual_hour_end

    def _build_holiday_set(self, years: set) -> set:
        if not _HOL_AVAILABLE:
            return set()
        result: set = set()
        for yr in years:
            try:
                result |= set(_hol.country_holidays(self.country, years=yr).keys())
            except Exception:
                pass
        return result

    def _get_unusual_users(self, df: pd.DataFrame) -> List:
        if "user_id" not in df.columns:
            return []
        counts = df["user_id"].value_counts(normalize=True)
        return list(counts[counts < self.top_user_pct].index)

    def _flag_same_day_reversals(self, df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["_date"]    = tmp["posting_date"].dt.date
        tmp["_abs_amt"] = tmp["amount"].abs().round(2)
        tmp["_key"]     = (
            tmp["_date"].astype(str) + "|"
            + tmp["_abs_amt"].astype(str) + "|"
            + tmp.get("account_code", pd.Series("", index=tmp.index)).astype(str)
        )
        # Group by key: flag if both positive and negative amounts exist
        def has_reversal(grp):
            return (grp["amount"] > 0).any() and (grp["amount"] < 0).any()

        reversal_keys = {
            key for key, grp in tmp.groupby("_key") if has_reversal(grp)
        }
        return tmp["_key"].isin(reversal_keys)

    def _flag_split_transactions(self, df: pd.DataFrame) -> pd.Series:
        """Detect potential invoice-splitting (just below approval threshold)."""
        just_below = self.materiality * 0.97
        result = pd.Series(False, index=df.index)
        if "vendor_id" not in df.columns:
            return result
        tmp = df.copy()
        tmp["_date"] = tmp["posting_date"].dt.date
        for (vendor, date), grp in tmp.groupby(["vendor_id", "_date"]):
            mask = (grp["amount"] >= just_below) & (grp["amount"] < self.materiality)
            if mask.sum() >= 2:
                result.loc[grp[mask].index] = True
        return result

    # ------------------------------------------------------------------
    # Main scoring method
    # ------------------------------------------------------------------
    def score_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        # Ensure datetime
        result["posting_date"] = pd.to_datetime(result["posting_date"], errors="coerce")

        for flag in _RISK_WEIGHTS:
            result[f"flag_{flag}"] = False

        unusual_users = self._get_unusual_users(result)
        reversal_flags = self._flag_same_day_reversals(result)
        split_flags = self._flag_split_transactions(result)

        # Pre-compute holiday set once for all years present in the data
        years_in_data = set(result["posting_date"].dropna().dt.year.astype(int))
        holiday_set = self._build_holiday_set(years_in_data)

        for idx in result.index:
            row = result.loc[idx]
            dt = row["posting_date"]
            if pd.isnull(dt):
                continue

            amount = float(row.get("amount", 0))
            hour = int(row.get("posting_hour", 12))
            account = str(row.get("account_code", ""))
            user = str(row.get("user_id", ""))
            je_type = str(row.get("je_type", "")).upper()

            if self._is_round_number(amount):
                result.at[idx, "flag_round_number"] = True

            if self._is_period_end(dt):
                result.at[idx, "flag_period_end"] = True

            if self._is_unusual_hour(hour):
                result.at[idx, "flag_unusual_hour"] = True

            if dt.dayofweek >= 5:
                result.at[idx, "flag_weekend"] = True

            if dt.date() in holiday_set:
                result.at[idx, "flag_public_holiday"] = True

            if reversal_flags.get(idx, False):
                result.at[idx, "flag_same_day_reversal"] = True

            if user in unusual_users:
                result.at[idx, "flag_unusual_user"] = True

            if abs(amount) > self.materiality * 5:
                result.at[idx, "flag_large_amount"] = True

            # Suspense / clearing accounts (9xxx in many ERP systems)
            if account.startswith("9") or "SUSPENSE" in account.upper() or "CLEARING" in account.upper():
                result.at[idx, "flag_suspense_account"] = True

            # Manual journal override flag
            if "MANUAL" in je_type or "OVERRIDE" in je_type:
                result.at[idx, "flag_manual_override"] = True

            # Intercompany
            if "IC" in account[:3].upper() or "INTER" in str(row.get("description", "")).upper():
                result.at[idx, "flag_intercompany"] = True

        result["flag_split_transaction"] = split_flags

        # Composite risk score
        result["risk_score"] = 0
        for flag, weight in _RISK_WEIGHTS.items():
            col = f"flag_{flag}"
            if col in result.columns:
                result["risk_score"] += result[col].astype(int) * weight
        result["risk_score"] = result["risk_score"].clip(0, 100)

        # Risk tier
        result["risk_level"] = pd.cut(
            result["risk_score"],
            bins=[-1, 15, 40, 70, 101],
            labels=["Low", "Medium", "High", "Critical"],
        )

        # Active flag count
        flag_cols = [c for c in result.columns if c.startswith("flag_")]
        result["flag_count"] = result[flag_cols].sum(axis=1)

        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def risk_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        flag_cols = [c for c in scored_df.columns if c.startswith("flag_")]
        rows = []
        for col in flag_cols:
            indicator = col.replace("flag_", "").replace("_", " ").title()
            n_flagged = int(scored_df[col].sum())
            pct = n_flagged / len(scored_df) * 100 if len(scored_df) > 0 else 0
            amount_flagged = scored_df.loc[scored_df[col], "amount"].abs().sum() if n_flagged > 0 else 0
            rows.append({
                "Indicator": indicator,
                "Count": n_flagged,
                "% of Entries": round(pct, 2),
                "Amount Flagged": round(amount_flagged, 2),
                "ISA Ref": _ISA_REFS.get(col.replace("flag_", ""), "ISA 240"),
            })
        return pd.DataFrame(rows).sort_values("Count", ascending=False)


_ISA_REFS: Dict[str, str] = {
    "same_day_reversal":  "ISA 240.A3",
    "round_number":       "ISA 240.A3 / Nigrini",
    "suspense_account":   "ISA 240.A33",
    "manual_override":    "ISA 240.A3",
    "period_end":         "ISA 240.A3",
    "unusual_hour":       "ISA 240.A3",
    "unusual_user":       "ISA 240.A3",
    "weekend":            "ISA 240.A3",
    "public_holiday":     "ISA 240.A3",
    "large_amount":       "ISA 315.A107",
    "intercompany":       "ISA 315.A85",
    "split_transaction":  "ISA 240.A3 / FCPA",
}
