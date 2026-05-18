"""
ML anomaly detection engine using Isolation Forest + SHAP explanations.

References:
  - Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest.
  - ISA 240 — The Auditor's Responsibilities Relating to Fraud
  - PCAOB AS 2401 — Consideration of Fraud in a Financial Statement Audit
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class AnomalyDetector:

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._encoders: Dict[str, LabelEncoder] = {}
        self._feature_names: List[str] = []
        self._explainer = None
        self._X_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        feats = pd.DataFrame(index=df.index)

        feats["log_amount"]    = np.log1p(df["amount"].abs())
        feats["day_of_month"]  = df["posting_date"].dt.day
        feats["day_of_week"]   = df["posting_date"].dt.dayofweek
        feats["month"]         = df["posting_date"].dt.month
        feats["posting_hour"]  = df["posting_hour"].astype(int) if "posting_hour" in df.columns else 12
        feats["is_weekend"]    = (df["posting_date"].dt.dayofweek >= 5).astype(int)
        feats["is_month_end"]  = (df["posting_date"].dt.day >= 26).astype(int)
        feats["is_q_end"]      = df["posting_date"].dt.month.isin([3, 6, 9, 12]).astype(int)

        cat_cols = ["account_type", "user_id", "entity"]
        for col in cat_cols:
            if col not in df.columns:
                feats[f"{col}_enc"] = 0
                continue
            le = self._encoders.get(col) if not fit else LabelEncoder()
            vals = df[col].fillna("UNKNOWN").astype(str)
            if fit:
                le.fit(vals)
                self._encoders[col] = le
            # Handle unseen labels at predict time
            known = set(le.classes_)
            vals_safe = vals.apply(lambda x: x if x in known else "UNKNOWN")
            # Ensure UNKNOWN is in classes
            if "UNKNOWN" not in known:
                le.classes_ = np.append(le.classes_, "UNKNOWN")
            feats[f"{col}_enc"] = le.transform(vals_safe)

        if "debit_credit" in df.columns:
            feats["is_credit"] = (df["debit_credit"] == "C").astype(int)
        else:
            feats["is_credit"] = 0

        if fit:
            self._feature_names = list(feats.columns)

        X = feats.reindex(columns=self._feature_names, fill_value=0).fillna(0).values
        if fit:
            return self._scaler.fit_transform(X)
        return self._scaler.transform(X)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = self._build_features(df, fit=True)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._X_train = X

        if _SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(self._model)
            except Exception:
                self._explainer = None
        return self

    # ------------------------------------------------------------------
    # Predict + explain
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        X = self._build_features(df, fit=False)
        raw_scores = self._model.decision_function(X)   # higher = more normal
        predictions = self._model.predict(X)            # -1 = anomaly

        # Invert and normalise to [0,1]: 1 = most anomalous
        s_min, s_max = raw_scores.min(), raw_scores.max()
        anomaly_score = 1.0 - (raw_scores - s_min) / (s_max - s_min + 1e-9)

        result = df.copy()
        result["anomaly_score"]      = np.round(anomaly_score, 4)
        result["is_anomaly"]         = predictions == -1
        result["anomaly_percentile"] = pd.Series(anomaly_score, index=df.index).rank(pct=True) * 100

        # SHAP explanations
        if _SHAP_AVAILABLE and self._explainer is not None:
            try:
                shap_vals = self._explainer.shap_values(X)
                abs_shap = np.abs(shap_vals)
                result["top_risk_driver"] = pd.Series(
                    [self._feature_names[i] for i in abs_shap.argmax(axis=1)],
                    index=df.index,
                )
                result["shap_top_value"] = abs_shap.max(axis=1).round(4)
            except Exception:
                result["top_risk_driver"] = "N/A"
                result["shap_top_value"]  = np.nan
        else:
            result["top_risk_driver"] = "N/A"
            result["shap_top_value"]  = np.nan

        return result

    # ------------------------------------------------------------------
    # Cluster anomalies
    # ------------------------------------------------------------------
    def cluster_anomalies(self, scored_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        anomalies = scored_df[scored_df["is_anomaly"]].copy()
        if len(anomalies) < max(n_clusters, 2):
            anomalies["cluster"] = 0
            anomalies["cluster_label"] = "Cluster A"
            return anomalies

        k = min(n_clusters, len(anomalies))
        feat_cols = ["anomaly_score"]
        if "log_amount" not in anomalies.columns:
            anomalies = anomalies.copy()
            anomalies["_log_amount"] = np.log1p(anomalies["amount"].abs())
            feat_cols.append("_log_amount")
        else:
            feat_cols.append("log_amount")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(anomalies[feat_cols].fillna(0))
        anomalies["cluster"] = labels
        anomalies["cluster_label"] = [f"Cluster {chr(65 + i)}" for i in labels]
        return anomalies

    # ------------------------------------------------------------------
    # Overlap analysis vs statistical flags
    # ------------------------------------------------------------------
    def overlap_analysis(
        self,
        ml_flags: pd.Series,
        stat_flags: pd.Series,
    ) -> Dict[str, int]:
        ml = ml_flags.astype(bool)
        st = stat_flags.astype(bool)
        return {
            "ml_flagged":   int(ml.sum()),
            "stat_flagged": int(st.sum()),
            "consensus":    int((ml & st).sum()),
            "ml_only":      int((ml & ~st).sum()),
            "stat_only":    int((~ml & st).sum()),
        }
