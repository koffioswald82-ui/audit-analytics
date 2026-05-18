"""
Vendor / counterparty risk analysis.

Covers:
  - Duplicate payment detection
  - Ghost vendor detection via fuzzy name matching
  - Vendor concentration analysis
  - Payment timing patterns
  - ISA 240.A3 — Fraud risk indicator alignment
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from fuzzywuzzy import fuzz, process
    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False


class VendorAnalyzer:

    def __init__(
        self,
        duplicate_window_days: int = 30,
        fuzzy_threshold: int = 85,
        concentration_top_n: int = 10,
    ):
        self.duplicate_window_days = duplicate_window_days
        self.fuzzy_threshold = fuzzy_threshold
        self.concentration_top_n = concentration_top_n

    # ------------------------------------------------------------------
    # Duplicate payment detection
    # ------------------------------------------------------------------
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag entries with same vendor + amount within `duplicate_window_days`.
        Returns only flagged rows with a `duplicate_group` column.
        """
        required = {"vendor_id", "amount", "posting_date"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        result = df.copy()
        result["posting_date"] = pd.to_datetime(result["posting_date"])
        result["_amt_rounded"] = result["amount"].abs().round(2)
        result["_dup_flag"] = False
        result["_dup_group"] = ""

        sorted_df = result.sort_values(["vendor_id", "_amt_rounded", "posting_date"])

        group_id = 0
        for (vendor, amt), grp in sorted_df.groupby(["vendor_id", "_amt_rounded"]):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("posting_date")
            dates = grp["posting_date"].values
            indices = grp.index.tolist()

            # Sliding window: if any two entries within window → duplicates
            for i in range(len(dates)):
                for j in range(i + 1, len(dates)):
                    delta = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days
                    if delta <= self.duplicate_window_days:
                        label = f"DUP-{group_id:04d}"
                        result.at[indices[i], "_dup_flag"] = True
                        result.at[indices[j], "_dup_flag"] = True
                        result.at[indices[i], "_dup_group"] = label
                        result.at[indices[j], "_dup_group"] = label
                        group_id += 1

        duplicates = result[result["_dup_flag"]].copy()
        duplicates.rename(columns={"_dup_group": "duplicate_group"}, inplace=True)
        return duplicates.drop(columns=["_dup_flag", "_amt_rounded"], errors="ignore")

    # ------------------------------------------------------------------
    # Ghost vendor detection
    # ------------------------------------------------------------------
    def detect_ghost_vendors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies vendor name pairs with high fuzzy similarity (potential ghost vendors).
        Returns a DataFrame of suspicious pairs with similarity score.
        """
        if not _FUZZY_AVAILABLE:
            return pd.DataFrame(columns=["vendor_a", "vendor_b", "similarity", "risk_note"])

        if "vendor_name" not in df.columns:
            return pd.DataFrame(columns=["vendor_a", "vendor_b", "similarity", "risk_note"])

        unique_names = df["vendor_name"].dropna().unique().tolist()
        if len(unique_names) < 2:
            return pd.DataFrame(columns=["vendor_a", "vendor_b", "similarity", "risk_note"])

        pairs = []
        checked = set()
        for name in unique_names:
            matches = process.extractBests(
                name,
                unique_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.fuzzy_threshold,
                limit=5,
            )
            for match_name, score in matches:
                if match_name == name:
                    continue
                key = tuple(sorted([name, match_name]))
                if key in checked:
                    continue
                checked.add(key)
                pairs.append({
                    "vendor_a": name,
                    "vendor_b": match_name,
                    "similarity": score,
                    "risk_note": (
                        "High similarity — possible duplicate/ghost vendor (ISA 240.A3)"
                        if score >= 92
                        else "Moderate similarity — verify vendor master data"
                    ),
                })

        return pd.DataFrame(pairs).sort_values("similarity", ascending=False)

    # ------------------------------------------------------------------
    # Concentration analysis
    # ------------------------------------------------------------------
    def concentration_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns top-N vendors by total spend with cumulative % of total.
        """
        if "vendor_name" not in df.columns or "amount" not in df.columns:
            return pd.DataFrame()

        spend = (
            df.groupby("vendor_name")["amount"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "total_amount", "count": "n_transactions"})
            .sort_values("total_amount", ascending=False)
        )
        total = spend["total_amount"].abs().sum()
        spend["pct_of_total"] = (spend["total_amount"].abs() / total * 100).round(2)
        spend["cumulative_pct"] = spend["pct_of_total"].cumsum().round(2)
        spend["avg_transaction"] = (spend["total_amount"] / spend["n_transactions"]).round(2)

        return spend.head(self.concentration_top_n).reset_index()

    # ------------------------------------------------------------------
    # Payment timing analysis
    # ------------------------------------------------------------------
    def payment_timing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Breakdown of payments by day of week and posting hour.
        """
        result = df.copy()
        result["posting_date"] = pd.to_datetime(result["posting_date"])
        result["day_name"] = result["posting_date"].dt.day_name()
        result["day_num"] = result["posting_date"].dt.dayofweek

        by_day = (
            result.groupby(["day_name", "day_num"])
            .agg(count=("amount", "size"), total=("amount", "sum"))
            .reset_index()
            .sort_values("day_num")
            .drop(columns="day_num")
        )

        if "posting_hour" in result.columns:
            result["hour_bucket"] = pd.cut(
                result["posting_hour"].astype(int),
                bins=[-1, 5, 8, 12, 17, 20, 24],
                labels=["00-05 Night", "06-08 Early", "09-12 Morning",
                        "13-17 Afternoon", "18-20 Evening", "21-24 Late Night"],
            )
            by_hour = (
                result.groupby("hour_bucket", observed=True)
                .agg(count=("amount", "size"), total=("amount", "sum"))
                .reset_index()
                .rename(columns={"hour_bucket": "time_window"})
            )
        else:
            by_hour = pd.DataFrame()

        return by_day, by_hour

    # ------------------------------------------------------------------
    # Full vendor report
    # ------------------------------------------------------------------
    def full_report(self, df: pd.DataFrame) -> Dict:
        return {
            "duplicates": self.detect_duplicates(df),
            "ghost_vendors": self.detect_ghost_vendors(df),
            "concentration": self.concentration_analysis(df),
        }
