"""
AnDi track-analysis adapter (Anomalous Diffusion challenge; Munoz-Gil et al. 2021/2024)
-- runs in the `andi` venv. Deep-learning inference of the anomalous-diffusion exponent
and the underlying diffusion model per trajectory.

Contract: tracks.csv (track_id, frame, x, y) in -> track_analysis.csv (track_id, alpha,
model) out.

BINDING POINT: AnDi methods use trained models (andi-datasets + a network). Wire
`_analyze()` to your AnDi model. The IO + analyze contract are done here.
"""

import argparse
import json

import numpy as np  # noqa: F401  (for the binding-point implementation)
import pandas as pd


def _analyze(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return one row per track including `track_id`. AnDi model binding point."""
    raise NotImplementedError(
        "AnDi model is the binding point: wire adapters/andi_analyze.py:_analyze to your "
        "trained AnDi network (andi-datasets). IO + the track_analysis contract are done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    _analyze(pd.read_csv(args.inp), json.loads(args.params)).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
