"""Unit tests — BenfordAnalyzer."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benford_analyzer import BenfordAnalyzer, BENFORD_FIRST


def _benford_sample(n: int = 5000, seed: int = 0) -> pd.Series:
    """Generate a Benford-distributed sample via log-uniform distribution."""
    rng = np.random.default_rng(seed)
    return pd.Series(10 ** rng.uniform(1, 6, n))


def _uniform_sample(n: int = 5000, seed: int = 1) -> pd.Series:
    """Uniform distribution — should NOT conform to Benford's Law."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(1000, 9999, n))


class TestBenfordAnalyzer:
    ba = BenfordAnalyzer()

    def test_conforming_data_low_risk(self):
        s = _benford_sample()
        result = self.ba.analyze(s, digit_type="first", alpha=0.05)
        assert result.isa_240_risk_level in ("LOW", "MEDIUM")
        assert result.n > 0

    def test_nonconforming_data_raises_risk(self):
        s = _uniform_sample()
        result = self.ba.analyze(s, digit_type="first", alpha=0.05)
        # Uniform data typically produces MEDIUM or HIGH risk
        assert result.isa_240_risk_level in ("MEDIUM", "HIGH")

    def test_mad_conformity_label(self):
        s = _benford_sample(n=10000)
        result = self.ba.analyze(s)
        assert result.mad_conformity in (
            "Close Conformity", "Acceptable Conformity",
            "Marginally Acceptable", "Nonconformity",
        )

    def test_chi_square_p_range(self):
        s = _benford_sample()
        result = self.ba.analyze(s)
        assert 0.0 <= result.chi_square_p <= 1.0

    def test_z_scores_non_negative(self):
        s = _benford_sample()
        result = self.ba.analyze(s)
        assert all(z >= 0 for z in result.z_scores.values())

    def test_red_flag_digits_subset_of_expected(self):
        s = _benford_sample()
        result = self.ba.analyze(s, digit_type="first")
        assert all(d in range(1, 10) for d in result.red_flag_digits)

    def test_insufficient_data_raises(self):
        s = pd.Series([100.0, 200.0, 300.0])
        with pytest.raises(ValueError, match="Insufficient data"):
            self.ba.analyze(s)

    def test_second_digit(self):
        s = _benford_sample(n=8000)
        result = self.ba.analyze(s, digit_type="second")
        assert result.digit_type == "second"
        assert all(d in range(0, 10) for d in result.observed.keys())

    def test_first_two_digits(self):
        s = _benford_sample(n=8000)
        result = self.ba.analyze(s, digit_type="first_two")
        assert result.digit_type == "first_two"

    def test_segment_analysis_returns_dataframe(self):
        df = pd.DataFrame({
            "amount": _benford_sample(2000),
            "entity": np.random.choice(["A", "B", "C"], 2000),
        })
        seg = self.ba.segment_analysis(df, "amount", "entity", min_count=50)
        assert isinstance(seg, pd.DataFrame)
        assert "ISA 240 Risk" in seg.columns

    def test_observed_sums_to_one(self):
        s = _benford_sample()
        result = self.ba.analyze(s)
        assert abs(sum(result.observed.values()) - 1.0) < 1e-6

    def test_min_amount_filter(self):
        s = pd.Series([0.5, 1.0, 1000.0, 50000.0] * 500)
        r_no_filter = self.ba.analyze(s, min_amount=0.01)
        r_filtered  = self.ba.analyze(s, min_amount=100.0)
        assert r_filtered.n < r_no_filter.n
