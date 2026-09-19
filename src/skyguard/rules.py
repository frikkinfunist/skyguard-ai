"""Layer 1: cheap rule-based checks — spikes, frozen sensors, dropouts."""

import numpy as np
import pandas as pd

SENSOR_COLS = ["temp", "pressure", "humidity"]


def rule_based_flags(df: pd.DataFrame, z_thresh: float = 4.0, freeze_window: int = 6) -> pd.DataFrame:
    """Flag spikes (rolling z-score), frozen values, and comms dropouts (NaN)."""
    flags = pd.DataFrame(index=df.index)
    for col in SENSOR_COLS:
        roll_mean = df[col].rolling(48, min_periods=10).mean()
        roll_std = df[col].rolling(48, min_periods=10).std()
        z = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        flags[f"{col}_spike"] = z.abs() > z_thresh

        frozen = (
            df[col].diff().rolling(freeze_window)
            .apply(lambda x: (x == 0).all(), raw=True)
            .fillna(0).astype(bool)
        )
        flags[f"{col}_frozen"] = frozen
        flags[f"{col}_dropout"] = df[col].isna()

    flags["rule_flag"] = flags.any(axis=1)
    return flags
