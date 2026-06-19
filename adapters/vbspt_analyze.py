"""
vbSPT track-analysis adapter (Persson et al., Nat. Methods 2013) -- runs in the `vbspt`
venv. Variational-Bayes hidden Markov model for diffusive-state segmentation of single
trajectories (recovers the number of states, their diffusion coefficients, and the
per-step state assignment).

Contract: tracks.csv (track_id, frame, x, y) in -> track_analysis.csv (track_id +
per-track dominant state / D) out.

BINDING POINT: vbSPT is distributed as MATLAB; wire `_analyze()` to your port (or a
Python re-implementation). The IO + analyze contract are done here.
"""

import argparse
import json

import numpy as np  # noqa: F401  (for the binding-point implementation)
import pandas as pd


def _analyze(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return one row per track including `track_id`. vbSPT HMM binding point."""
    raise NotImplementedError(
        "vbSPT HMM is the binding point: wire adapters/vbspt_analyze.py:_analyze to your "
        "vbSPT port. IO + the track_analysis contract are done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    _analyze(pd.read_csv(args.inp), json.loads(args.params)).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
