"""
FRC resolution and NeNA precision metrics.

Run: python -m pytest labflow/tests/test_metrics.py
(or:  python labflow/tests/test_metrics.py)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.stages.metrics import _frc, _nena


def test_nena_recovers_precision():
    rng = np.random.default_rng(0)
    sigma = 15.0                                    # injected per-coordinate precision (nm)
    centers = rng.uniform(0, 10000, (60, 2))        # isolated molecules
    xs, ys, fs = [], [], []
    for f in range(25):                             # each reappears across frames + jitter
        for c in centers:
            xs.append(c[0] + rng.normal(0, sigma))
            ys.append(c[1] + rng.normal(0, sigma))
            fs.append(f)
    est = _nena(np.array(xs), np.array(ys), np.array(fs))
    assert 0.6 * sigma < est < 1.6 * sigma, f"NeNA precision {est:.1f}, injected {sigma}"


def test_frc_finer_localization_gives_better_resolution():
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 5000, (1500, 2))          # a fixed structure

    def resolution(jitter):
        P = np.repeat(base, 3, axis=0) + rng.normal(0, jitter, (base.shape[0] * 3, 2))
        return _frc(P[:, 0], P[:, 1], render_nm=10)[2]

    fine = resolution(5.0)
    coarse = resolution(40.0)
    assert np.isfinite(fine), "FRC resolution should be finite on structured data"
    assert fine < coarse, f"finer localization should resolve better: {fine:.0f} vs {coarse:.0f}"


if __name__ == "__main__":
    test_nena_recovers_precision()
    test_frc_finer_localization_gives_better_resolution()
    print("metrics (FRC + NeNA) tests passed")
