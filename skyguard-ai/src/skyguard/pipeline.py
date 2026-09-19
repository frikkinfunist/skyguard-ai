"""Combine all layers into one confidence-weighted alert per reading."""

import numpy as np
import pandas as pd

from .data import make_station_data
from .rules import rule_based_flags
from .baseline import seasonal_residuals
from .detector import multivariate_scores


def build_alerts(df: pd.DataFrame, rule_flags: pd.DataFrame, mv_scores: pd.Series, mv_flag: pd.Series) -> pd.DataFrame:
    conf = (mv_scores.max() - mv_scores) / (mv_scores.max() - mv_scores.min())
    out = pd.DataFrame({
        "rule_flag": rule_flags["rule_flag"],
        "mv_flag": mv_flag.reindex(df.index).fillna(False),
        "mv_confidence": conf.reindex(df.index).fillna(0),
    })
    out["alert"] = out["rule_flag"] | out["mv_flag"]
    out["severity"] = np.where(
        out["rule_flag"] & out["mv_flag"], "high",
        np.where(out["alert"], "medium", "none"),
    )
    return out


def run_pipeline(days: int = 90, freq_min: int = 30, seed: int = 42):
    """End-to-end run for one station. Returns (df, alerts)."""
    df = make_station_data(days=days, freq_min=freq_min, seed=seed)
    rules = rule_based_flags(df)
    resid = seasonal_residuals(df)
    mv_scores, mv_flag = multivariate_scores(resid)
    alerts = build_alerts(df, rules, mv_scores, mv_flag)
    return df, alerts
