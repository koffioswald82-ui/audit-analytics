"""Unit tests — AnomalyDetector and JournalEntryAnalyzer."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.anomaly_detector import AnomalyDetector
from src.journal_entry_analyzer import JournalEntryAnalyzer


def _make_gl(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365, n)]
    return pd.DataFrame({
        "je_id":        [f"JE-{i:05d}" for i in range(n)],
        "entity":       rng.choice(["ACME", "TechFlow", "FinServ"], n),
        "posting_date": dates,
        "account_code": rng.choice(["1100", "4100", "6100", "6300"], n),
        "account_type": rng.choice(["Asset", "Revenue", "Expense"], n),
        "amount":       10 ** rng.uniform(2, 6, n),
        "debit_credit": rng.choice(["D", "C"], n),
        "user_id":      rng.choice(["j.martin", "s.chen", "SYS_TEMP_01"], n),
        "posting_hour": rng.integers(0, 24, n),
        "je_type":      rng.choice(["Automated", "Manual"], n),
        "vendor_id":    rng.choice(["V001", "V002", "V003"], n),
        "vendor_name":  rng.choice(["Apex Consulting", "Global IT", "Premier Supply"], n),
        "description":  ["Standard posting"] * n,
    })


class TestAnomalyDetector:
    def test_fit_predict_returns_correct_columns(self):
        df = _make_gl(300)
        det = AnomalyDetector(contamination=0.05, n_estimators=50)
        det.fit(df)
        result = det.predict(df)
        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns
        assert "anomaly_percentile" in result.columns

    def test_anomaly_rate_near_contamination(self):
        df = _make_gl(500)
        contamination = 0.10
        det = AnomalyDetector(contamination=contamination, n_estimators=50)
        det.fit(df)
        result = det.predict(df)
        actual_rate = result["is_anomaly"].mean()
        assert abs(actual_rate - contamination) < 0.05

    def test_anomaly_score_bounded(self):
        df = _make_gl(200)
        det = AnomalyDetector(contamination=0.05, n_estimators=50)
        det.fit(df)
        result = det.predict(df)
        assert result["anomaly_score"].between(0, 1).all()

    def test_predict_before_fit_raises(self):
        det = AnomalyDetector()
        df = _make_gl(100)
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(df)

    def test_cluster_anomalies(self):
        df = _make_gl(300)
        det = AnomalyDetector(contamination=0.10, n_estimators=50)
        det.fit(df)
        scored = det.predict(df)
        clusters = det.cluster_anomalies(scored, n_clusters=3)
        assert "cluster" in clusters.columns
        assert len(clusters) == scored["is_anomaly"].sum()

    def test_overlap_analysis(self):
        det = AnomalyDetector()
        ml_flags   = pd.Series([True, False, True, True, False])
        stat_flags = pd.Series([False, False, True, False, True])
        result = det.overlap_analysis(ml_flags, stat_flags)
        assert result["ml_flagged"] == 3
        assert result["stat_flagged"] == 2
        assert result["consensus"] == 1


class TestJournalEntryAnalyzer:
    def test_score_returns_risk_score_column(self):
        df = _make_gl(200)
        jea = JournalEntryAnalyzer(materiality=10_000)
        result = jea.score_entries(df)
        assert "risk_score" in result.columns
        assert "risk_level" in result.columns

    def test_risk_score_bounded(self):
        df = _make_gl(300)
        jea = JournalEntryAnalyzer(materiality=10_000)
        result = jea.score_entries(df)
        assert result["risk_score"].between(0, 100).all()

    def test_round_number_flagged(self):
        df = pd.DataFrame({
            "je_id": ["JE-001"],
            "posting_date": [pd.Timestamp("2023-06-15")],
            "amount": [10_000.00],
            "account_code": ["6300"],
            "user_id": ["j.martin"],
            "posting_hour": [10],
            "je_type": ["Manual"],
            "description": ["Test"],
        })
        jea = JournalEntryAnalyzer(materiality=50_000)
        result = jea.score_entries(df)
        assert result.iloc[0]["flag_round_number"] is True

    def test_weekend_flagged(self):
        # 2023-07-08 is a Saturday
        df = pd.DataFrame({
            "je_id": ["JE-002"],
            "posting_date": [pd.Timestamp("2023-07-08")],
            "amount": [5_000.00],
            "account_code": ["4100"],
            "user_id": ["j.martin"],
            "posting_hour": [10],
            "je_type": ["Manual"],
            "description": ["Test weekend"],
        })
        jea = JournalEntryAnalyzer()
        result = jea.score_entries(df)
        assert result.iloc[0]["flag_weekend"] is True

    def test_unusual_hour_flagged(self):
        df = pd.DataFrame({
            "je_id": ["JE-003"],
            "posting_date": [pd.Timestamp("2023-06-15")],
            "amount": [3_000.00],
            "account_code": ["6100"],
            "user_id": ["j.martin"],
            "posting_hour": [2],    # 2am
            "je_type": ["Manual"],
            "description": ["Late night"],
        })
        jea = JournalEntryAnalyzer(unusual_hour_start=20, unusual_hour_end=6)
        result = jea.score_entries(df)
        assert result.iloc[0]["flag_unusual_hour"] is True

    def test_risk_summary_returns_dataframe(self):
        df = _make_gl(200)
        jea = JournalEntryAnalyzer()
        scored = jea.score_entries(df)
        summary = jea.risk_summary(scored)
        assert isinstance(summary, pd.DataFrame)
        assert "Indicator" in summary.columns
        assert "Count" in summary.columns
