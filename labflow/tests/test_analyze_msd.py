"""
MSD diffusion analysis recovers a known diffusion coefficient and Brownian exponent.

Run: python -m pytest labflow/tests/test_analyze_msd.py
(or:  python labflow/tests/test_analyze_msd.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.stages.analyze_msd import run


def _brownian(n_tracks=80, n_steps=40, D=50.0, seed=0):
    """2D Brownian tracks with known D (per-coordinate step std = sqrt(2 D), dt=1)."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D)
    rows = []
    for tid in range(n_tracks):
        start = rng.uniform(0, 10000, 2)
        pos = start + np.cumsum(rng.normal(0, sigma, (n_steps, 2)), axis=0)
        for f in range(n_steps):
            rows.append({"track_id": tid, "frame": f, "x": pos[f, 0], "y": pos[f, 1]})
    return pd.DataFrame(rows)


def test_recovers_D_and_brownian_alpha():
    D_true = 50.0
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "tracks.csv")
        _brownian(D=D_true).to_csv(inp, index=False)
        out = os.path.join(tmp, "analysis.csv")
        run(input_csv=inp, output_csv=out, params={"max_lag": 4, "min_length": 5})

        a = pd.read_csv(out)
        assert "track_id" in a.columns and len(a) >= 70
        D_med = a["diffusion_coefficient"].median()
        assert 0.5 * D_true < D_med < 2.0 * D_true, f"recovered D={D_med:.1f}, true={D_true}"
        assert 0.6 < a["alpha"].median() < 1.4, f"Brownian alpha should be ~1, got {a['alpha'].median():.2f}"


if __name__ == "__main__":
    test_recovers_D_and_brownian_alpha()
    print("analyze_msd test passed")
