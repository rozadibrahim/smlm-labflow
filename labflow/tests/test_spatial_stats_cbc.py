"""
CBC (coordinate-based colocalization) distinguishes colocalized from segregated channels.

Run: python -m pytest labflow/tests/test_spatial_stats_cbc.py
(or:  python labflow/tests/test_spatial_stats_cbc.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.stages.spatial_stats import run


def _write(tmp, name, P):
    path = os.path.join(tmp, name)
    pd.DataFrame({"frame": range(len(P)), "x": P[:, 0], "y": P[:, 1]}).to_csv(path, index=False)
    return path


def _mean_cbc(a_csv, b_csv, out):
    run(input_csv=a_csv, output_csv=out,
        params={"metric": "cbc", "channel2": b_csv, "r_max": 300, "n_bins": 20})
    return pd.read_csv(out)["cbc"].mean()


def test_cbc_coloc_vs_segregated():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        A = rng.normal((1000, 1000), 50, (200, 2))
        B_coloc = rng.normal((1000, 1000), 50, (200, 2))    # same region as A
        B_seg = rng.normal((6000, 6000), 50, (200, 2))      # far from A

        a = _write(tmp, "A.csv", A)
        coloc = _mean_cbc(a, _write(tmp, "Bc.csv", B_coloc), os.path.join(tmp, "c.csv"))
        seg = _mean_cbc(a, _write(tmp, "Bs.csv", B_seg), os.path.join(tmp, "s.csv"))

        assert coloc > seg, f"colocalized CBC {coloc:.3f} should exceed segregated {seg:.3f}"
        assert coloc > 0.2, f"colocalized CBC should be clearly positive, got {coloc:.3f}"


if __name__ == "__main__":
    test_cbc_coloc_vs_segregated()
    print("CBC spatial-stats test passed")
