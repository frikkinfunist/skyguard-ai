# SkyGuard AI

Real-time anomaly detection for Automatic Weather Station (AWS) sensor
streams — temperature, pressure, humidity. Built for SIH26073 (Ministry of
Earth Sciences / IMD).

Sensors fail silently — a stuck value, a comms dropout, a physically
impossible spike — and bad data quietly corrupts downstream forecasts and
disaster alerts. SkyGuard AI is a lightweight layer that catches these
faults per station, in real time, before they propagate.

## Pipeline

Three layers, each cheap and each independently useful:

1. **Rule-based checks** (`skyguard/rules.py`) — rolling z-score for spikes,
   a frozen-value detector, and dropout (`NaN`) detection. Near-zero cost,
   catches the obvious cases instantly.
2. **Seasonal baseline** (`skyguard/baseline.py`) — a per-station expected
   value keyed on (week-of-year, hour-of-day), so the same absolute reading
   can be normal in one season and anomalous in another. India's climate
   varies enough by region that a single global baseline would misfire
   constantly — this stays per-station.
3. **Multivariate consistency scoring** (`skyguard/detector.py`) — an
   Isolation Forest over the three seasonal residuals jointly, to catch
   combinations that look fine sensor-by-sensor but are physically
   implausible together (e.g. a pressure drop with no matching humidity
   rise).

`skyguard/pipeline.py` combines all three into one alert per reading, with
severity `high` when both the rule layer and the multivariate layer agree,
and `medium` when only one does — this two-tier severity is what keeps the
alert volume from becoming noise.

## Why this architecture

- **Isolation Forest over a deep model** — unsupervised (no labelled fault
  data needed, which is scarce for real AWS faults), trains in seconds,
  and stays interpretable. With only 3 input features, a large neural
  model is unnecessary complexity.
- **Layered, not end-to-end** — each layer works and can be demoed on its
  own; the multivariate layer being unfinished doesn't block the rule
  layer from already catching most faults.
- **Per-station baseline, not global** — see `baseline.py`; a Rajasthan
  station and a Kerala station don't share a "normal."

## Project structure

```
skyguard-ai/
├── src/skyguard/
│   ├── data.py        # synthetic AWS data generator (stand-in for a real feed)
│   ├── rules.py        # Layer 1 — rule-based checks
│   ├── baseline.py      # Layer 2 — per-station seasonal residual
│   ├── detector.py      # Layer 3 — Isolation Forest multivariate scoring
│   └── pipeline.py      # combines layers into confidence-weighted alerts
├── examples/run_single_station.py   # end-to-end run + diagnostic plot
├── tests/test_pipeline.py            # sanity tests
└── outputs/                          # generated plots land here
```

## Quickstart

```bash
pip install -r requirements.txt
python examples/run_single_station.py
```

This runs the full pipeline on synthetic data with injected faults (spikes,
a frozen sensor stretch, comms dropouts), prints a summary + sample flagged
rows, and saves a diagnostic plot to `outputs/station_pipeline_output.png`.

Run tests:

```bash
python tests/test_pipeline.py
```

## Roadmap / not yet built

- **Explainability**: attach SHAP (`KernelExplainer` on `decision_function`,
  or `TreeExplainer` on the underlying trees) so every alert ships with a
  root-cause reason, not just a flag.
- **Serving**: wrap `run_pipeline` in a FastAPI real-time inference service.
- **Dashboard**: Streamlit/Plotly view of live station health.
- **Validation on real data**: currently validated only on synthetic
  injected faults; needs real IMD/AWS historical data with genuine fault
  incidents.
- **Edge deployment**: TensorFlow Lite Micro on ESP32 for offline-first,
  low-connectivity stations.
- **Climate-zone fallback baseline**: for new stations without enough
  history to learn their own seasonal pattern yet.

## License

MIT — see `LICENSE`.
