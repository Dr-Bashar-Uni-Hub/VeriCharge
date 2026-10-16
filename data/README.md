# Data

## Raw data
The study uses **ACN-Data** from the Caltech Adaptive Charging Network site (Lee, Li & Low, 2019). The original pipeline (`notebooks/01_VeriCharge_pipeline.ipynb`) reads the static ACN-Data archive and builds an energy-preserving 15-min site-load series. Raw session data are **not** redistributed in this repository; please obtain them from the ACN-Data providers and follow their terms of use. If you use the ACN-Data API, keep your API token private and never commit it.

## Frozen artifacts (GitHub release)
`vericharge_artifacts.zip` (attached to the repository release) contains the derived inputs needed by `analysis/` and the demo notebook:

- `data/forecasting/` — 15-min supervised forecasting arrays (24-h input windows, 15/30/60-min targets) and chronological role metadata;
- `results/` — frozen forecaster outputs, reliability signals (U, R, S), trust features, frozen trust models and thresholds, deep-ensemble predictions (B1) and fault-injection outputs (B2).

Unzip it into `./artifacts` at the repository root.
