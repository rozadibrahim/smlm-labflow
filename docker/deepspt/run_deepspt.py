#!/usr/bin/env python
"""
DeepSPT diffusional-fingerprinting runner -- runs INSIDE the smlm-labflow/deepspt image
(Jacobsen et al. 2024), invoked by labflow's `runtime: docker` analyze method.

Contract: tracks.csv (track_id, frame, x, y) in -> track_analysis.csv (track_id +
diffusional fingerprint / diffusion-type classification) out, the labflow `analyze`
stage output.

BINDING POINT: DeepSPT is a deep-learning model (temporal-segmentation + diffusional
fingerprinting + classifier) whose API is notebook-driven and needs trained weights.
The IO + the analyze contract are complete -- wire `_analyze()` to the DeepSPT model in
your image (hatzakislab/DeepSPT). It raises a clear message until then (no fabricated API).
"""

import argparse
import json

import pandas as pd


def _analyze(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return one row per track including `track_id`. DeepSPT model binding point."""
    raise NotImplementedError(
        "DeepSPT inference is the binding point: wire docker/deepspt/run_deepspt.py:_analyze "
        "to the DeepSPT fingerprinting model + weights (hatzakislab/DeepSPT). IO + the "
        "track_analysis contract are done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    result = _analyze(pd.read_csv(args.inp), json.loads(args.params))
    result.to_csv(args.out, index=False)
    print(f"deepspt: fingerprinted {len(result)} tracks -> {args.out}")


if __name__ == "__main__":
    main()
