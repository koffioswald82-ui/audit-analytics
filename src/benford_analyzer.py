"""
Benford's Law conformity testing engine.

References:
  - Nigrini, M.J. (2012). Benford's Law: Applications for Forensic Accounting,
    Auditing, and Fraud Detection. Wiley.
  - ISA 240 — The Auditor's Responsibilities Relating to Fraud in an Audit
  - ISA 500 — Audit Evidence
  - ISA 315 — Identifying and Assessing the Risks of Material Misstatement
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


BENFORD_FIRST: Dict[int, float] = {
    1: 0.30103, 2: 0.17609, 3: 0.12494, 4: 0.09691,
    5: 0.07918, 6: 0.06695, 7: 0.05799, 8: 0.05115, 9: 0.04576,
}

BENFORD_SECOND: Dict[int, float] = {
    0: 0.11968, 1: 0.11389, 2: 0.10882, 3: 0.10433, 4: 0.10031,
    5: 0.09668, 6: 0.09337, 7: 0.09035, 8: 0.08757, 9: 0.08500,
}

BENFORD_FIRST_TWO: Dict[int, float] = {
    d: np.log10(1 + 1 / d) for d in range(10, 100)
}

# Nigrini (2012) MAD conformity thresholds
MAD_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "first":     {"close": 0.006,  "acceptable": 0.012, "marginal": 0.015},
    "second":    {"close": 0.008,  "acceptable": 0.010, "marginal": 0.012},
    "first_two": {"close": 0.0012, "acceptable": 0.0018, "marginal": 0.0022},
}


@dataclass
class BenfordResult:
    digit_type: str
    n: int
    observed: Dict[int, float]
    expected: Dict[int, float]
    chi_square: float
    chi_square_p: float
    chi_square_critical: float
    mad: float
    mad_conformity: str
    z_scores: Dict[int, float]
    red_flag_digits: List[int]
    isa_240_risk_level: str
    isa_240_comment: str
    ks_statistic: float
    ks_p_value: float


class BenfordAnalyzer:

    def _get_expected(self, digit_type: str) -> Dict[int, float]:
        mapping = {
            "first": BENFORD_FIRST,
            "second": BENFORD_SECOND,
            "first_two": BENFORD_FIRST_TWO,
        }
        if digit_type not in mapping:
            raise ValueError(f"Unknown digit_type: {digit_type!r}. Use 'first', 'second', or 'first_two'.")
        return mapping[digit_type]

    def _extract_digits(self, series: pd.Series, digit_type: str) -> pd.Series:
        # Represent each number as a string of significant digits
        s = series.abs().replace(0, np.nan).dropna()
        # Format to avoid scientific notation
        s_str = s.apply(lambda x: f"{x:.6f}".replace(".", "").lstrip("0"))
        s_str = s_str[s_str.str.len() > 0]

        if digit_type == "first":
            return s_str.str[0].astype(int)
        elif digit_type == "second":
            s_str = s_str[s_str.str.len() >= 2]
            return s_str.str[1].astype(int)
        elif digit_type == "first_two":
            s_str = s_str[s_str.str.len() >= 2]
            return s_str.str[:2].astype(int)
        raise ValueError(f"Unknown digit_type: {digit_type!r}")

    def _mad(self, observed: Dict[int, float], expected: Dict[int, float]) -> float:
        keys = sorted(set(observed) & set(expected))
        if not keys:
            return 0.0
        return float(np.mean([abs(observed.get(k, 0.0) - expected[k]) for k in keys]))

    def _interpret_mad(self, mad: float, digit_type: str) -> str:
        t = MAD_THRESHOLDS.get(digit_type, MAD_THRESHOLDS["first"])
        if mad <= t["close"]:
            return "Close Conformity"
        elif mad <= t["acceptable"]:
            return "Acceptable Conformity"
        elif mad <= t["marginal"]:
            return "Marginally Acceptable"
        return "Nonconformity"

    def _chi_square(
        self, observed: Dict[int, float], expected: Dict[int, float], n: int
    ) -> Tuple[float, float, float]:
        keys = sorted(set(observed) & set(expected))
        obs = np.array([observed.get(k, 0.0) * n for k in keys], dtype=float)
        exp = np.array([expected[k] * n for k in keys], dtype=float)
        exp = np.clip(exp, 1e-9, None)
        chi2 = float(np.sum((obs - exp) ** 2 / exp))
        df = len(keys) - 1
        p_val = float(1 - stats.chi2.cdf(chi2, df))
        critical_95 = float(stats.chi2.ppf(0.95, df))
        return chi2, p_val, critical_95

    def _z_scores(
        self, observed: Dict[int, float], expected: Dict[int, float], n: int
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        for d, p_e in expected.items():
            p_o = observed.get(d, 0.0)
            denom = np.sqrt(p_e * (1 - p_e) / n)
            if denom < 1e-12:
                result[d] = 0.0
            else:
                z = (abs(p_o - p_e) - 1.0 / (2 * n)) / denom
                result[d] = float(max(z, 0.0))
        return result

    def _ks_test(
        self, observed: Dict[int, float], expected: Dict[int, float]
    ) -> Tuple[float, float]:
        keys = sorted(set(observed) & set(expected))
        obs_cdf = np.cumsum([observed.get(k, 0.0) for k in keys])
        exp_cdf = np.cumsum([expected[k] for k in keys])
        ks = float(np.max(np.abs(obs_cdf - exp_cdf)))
        n_approx = len(keys)
        p_approx = float(np.exp(-2 * (ks * np.sqrt(n_approx)) ** 2)) if n_approx > 0 else 1.0
        return ks, min(p_approx, 1.0)

    def analyze(
        self,
        amounts: pd.Series,
        digit_type: str = "first",
        min_amount: float = 1.0,
        filter_positive: bool = True,
        alpha: float = 0.05,
    ) -> BenfordResult:
        s = amounts.copy().dropna()
        if filter_positive:
            s = s[s > 0]
        s = s[s.abs() >= min_amount]

        digits = self._extract_digits(s, digit_type)
        n = len(digits)
        if n < 10:
            raise ValueError(f"Insufficient data: {n} observations (minimum 10 required).")

        expected = self._get_expected(digit_type)
        counts = digits.value_counts()
        observed = {d: float(counts.get(d, 0) / n) for d in expected}

        chi2, p_chi2, chi2_crit = self._chi_square(observed, expected, n)
        mad = self._mad(observed, expected)
        mad_conf = self._interpret_mad(mad, digit_type)
        z_sc = self._z_scores(observed, expected, n)
        ks_stat, ks_p = self._ks_test(observed, expected)

        red_flags = [d for d, z in z_sc.items() if z > 1.96]

        t = MAD_THRESHOLDS.get(digit_type, MAD_THRESHOLDS["first"])
        nonconform = mad > t["marginal"]
        stat_sig = p_chi2 < alpha

        if nonconform and stat_sig:
            risk = "HIGH"
            comment = (
                "Significant non-conformity detected. Expand substantive testing. "
                "Investigate red-flag digits for fabricated or manipulated amounts "
                "(ISA 240.A4, ISA 240.A26). Consider fraud risk factors per ISA 240.24."
            )
        elif stat_sig or nonconform:
            risk = "MEDIUM"
            comment = (
                "Moderate deviation from Benford's Law. Perform additional analytical "
                "procedures and management inquiry. Assess risk of material misstatement "
                "(ISA 315.A107, ISA 500.A14)."
            )
        else:
            risk = "LOW"
            comment = (
                "Data conforms to Benford's Law. No statistically significant deviations. "
                "Standard audit procedures apply (ISA 500.A1)."
            )

        return BenfordResult(
            digit_type=digit_type,
            n=n,
            observed=observed,
            expected=expected,
            chi_square=chi2,
            chi_square_p=p_chi2,
            chi_square_critical=chi2_crit,
            mad=mad,
            mad_conformity=mad_conf,
            z_scores=z_sc,
            red_flag_digits=red_flags,
            isa_240_risk_level=risk,
            isa_240_comment=comment,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
        )

    def segment_analysis(
        self,
        df: pd.DataFrame,
        amount_col: str,
        segment_col: str,
        digit_type: str = "first",
        min_count: int = 50,
        alpha: float = 0.05,
        min_amount: float = 1.0,
    ) -> pd.DataFrame:
        rows = []
        for seg, grp in df.groupby(segment_col):
            if len(grp) < min_count:
                continue
            try:
                r = self.analyze(
                    grp[amount_col],
                    digit_type=digit_type,
                    alpha=alpha,
                    min_amount=min_amount,
                )
                rows.append({
                    "Segment": seg,
                    "N": r.n,
                    "Chi²": round(r.chi_square, 2),
                    "p-value": round(r.chi_square_p, 4),
                    "MAD": round(r.mad, 5),
                    "Conformity": r.mad_conformity,
                    "Red Flag Digits": ", ".join(str(d) for d in r.red_flag_digits) or "—",
                    "ISA 240 Risk": r.isa_240_risk_level,
                })
            except ValueError:
                pass
        return pd.DataFrame(rows)
