"""Synthetic Automatic Weather Station (AWS) data — stand-in for a real feed."""

import numpy as np
import pandas as pd


def make_station_data(days: int = 90, freq_min: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate one station's temp/pressure/humidity series with injected faults.

    Faults injected: temperature spikes, a frozen-humidity stretch, and
    pressure comms dropouts (NaN) — used to sanity-check the detectors.
    """
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // freq_min
    t = pd.date_range("2026-01-01", periods=n, freq=f"{freq_min}min")
    hour = t.hour + t.minute / 60
    doy = t.dayofyear

    temp = (22 + 8 * np.sin(2 * np.pi * (hour - 6) / 24)
            + 4 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 0.6, n))
    pressure = 1013 + 3 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 0.8, n)
    humidity = (60 - 15 * np.sin(2 * np.pi * (hour - 6) / 24)
                + 10 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 3, n))
    df = pd.DataFrame({"time": t, "temp": temp, "pressure": pressure, "humidity": humidity})

    spike_idx = rng.choice(n, 5, replace=False)
    df.loc[spike_idx, "temp"] += rng.choice([-1, 1], 5) * rng.uniform(15, 25, 5)

    freeze_start = rng.integers(0, n - 20)
    df.loc[freeze_start:freeze_start + 15, "humidity"] = df.loc[freeze_start, "humidity"]

    drop_idx = rng.choice(n, 8, replace=False)
    df.loc[drop_idx, "pressure"] = np.nan

    return df.set_index("time")
