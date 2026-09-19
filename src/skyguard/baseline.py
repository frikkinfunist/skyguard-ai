"""Layer 2: per-station seasonal baseline — residual = actual - expected.

Expected value is the median for that station at that (week-of-year, hour)
bucket, so the same absolute reading can be normal in one season/time and
anomalous in another.
"""

import pandas as pd

SENSOR_COLS = ["temp", "pressure", "humidity"]


def seasonal_residuals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour"] = d.index.hour
    d["doy_bucket"] = d.index.dayofyear // 7  # weekly bucket, smooths noise

    resid = pd.DataFrame(index=df.index)
    for col in SENSOR_COLS:
        baseline = d.groupby(["doy_bucket", "hour"])[col].transform("median")
        resid[col] = d[col] - baseline
    return resid.fillna(0)
