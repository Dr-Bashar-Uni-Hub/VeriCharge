# Analysis scripts

Run from the repository root (see the main README for order and timing).
`common.py` loads the frozen artifacts from `$VERICHARGE_ARTIFACTS` (default `./artifacts`) and defines shared gating utilities;
all fitted gates use TrustFit only and 70%-coverage thresholds from TrustCalibration, exactly as in the frozen protocol.
These analyses were added after the FinalTest results of the original policy were known and are reported as post hoc in the paper.
