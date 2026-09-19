"""Run the pipeline for one station and save a diagnostic plot.

Usage:
    python examples/run_single_station.py
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from skyguard import run_pipeline  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    df, alerts = run_pipeline()

    print(alerts["severity"].value_counts())
    print("\nSample flagged readings:")
    print(alerts[alerts["alert"]].join(df).head(10))

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, col in enumerate(["temp", "pressure", "humidity"]):
        ax[i].plot(df.index, df[col], lw=0.7)
        flagged = alerts[alerts["alert"]].index
        ax[i].scatter(flagged, df.loc[flagged, col], color="red", s=15, zorder=5, label="alert")
        ax[i].set_ylabel(col)
        ax[i].legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "station_pipeline_output.png")
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
