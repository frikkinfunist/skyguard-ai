import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from skyguard import run_pipeline  # noqa: E402


def test_pipeline_runs_and_flags_injected_faults():
    df, alerts = run_pipeline(days=30, seed=1)

    assert len(df) == len(alerts)
    assert alerts["severity"].isin(["none", "medium", "high"]).all()
    # the synthetic generator always injects faults, so some alerts must fire
    assert alerts["alert"].sum() > 0


def test_severity_high_implies_both_layers_agree():
    _, alerts = run_pipeline(days=30, seed=1)
    high = alerts[alerts["severity"] == "high"]
    assert (high["rule_flag"] & high["mv_flag"]).all()


if __name__ == "__main__":
    test_pipeline_runs_and_flags_injected_faults()
    test_severity_high_implies_both_layers_agree()
    print("All tests passed.")
