"""
Regression tests for the RCC drift backend.

Run: python -m pytest driftcorr/tests/test_rcc.py
(or:  python driftcorr/tests/test_rcc.py  for a quick standalone check)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from driftcorr.core import apply_drift
from driftcorr.rcc import estimate_drift_rcc


def _make_locs(seed, n_emit, n_frames, inj_dx, inj_dy, structured):
    rng = np.random.default_rng(seed)
    if structured:
        pts = []
        for _ in range(12):                       # filaments + clusters
            x0, y0 = rng.uniform(2000, 26000, 2)
            ang, length = rng.uniform(0, np.pi), rng.uniform(3000, 9000)
            t = rng.uniform(0, length, n_emit // 12)
            pts.append(np.column_stack([
                x0 + t * np.cos(ang) + rng.normal(0, 40, t.size),
                y0 + t * np.sin(ang) + rng.normal(0, 40, t.size),
            ]))
        emit = np.vstack(pts)
    else:
        emit = np.column_stack([rng.uniform(0, 20000, n_emit),
                                rng.uniform(0, 20000, n_emit)])
    rows = []
    for fr in range(n_frames):
        k = rng.integers(len(emit), size=200)
        rows.append(np.column_stack([
            np.full(k.size, fr),
            emit[k, 0] + inj_dx[fr] + rng.normal(0, 8, k.size),
            emit[k, 1] + inj_dy[fr] + rng.normal(0, 8, k.size),
        ]))
    arr = np.vstack(rows)
    return pd.DataFrame({"frame": arr[:, 0].astype(int), "x": arr[:, 1], "y": arr[:, 2]})


def _rms_after_gauge(rec, inj):
    rec, inj = rec - rec.mean(), inj - inj.mean()
    return float(np.sqrt(np.mean((rec - inj) ** 2)))


def _recover(structured):
    n_frames = 2000
    f = np.arange(n_frames)
    inj_dx = 250 * f / n_frames + 40 * np.sin(2 * np.pi * f / n_frames)
    inj_dy = -180 * f / n_frames
    locs = _make_locs(7, 3000, n_frames, inj_dx, inj_dy, structured)
    est = estimate_drift_rcc(locs, units="nm",
                             params={"n_time_bins": 20, "render_nm": 20,
                                     "neighbor_span": 6, "max_drift_nm": 1000})
    rec_dx = np.interp(f, est.frames, est.dx)
    rec_dy = np.interp(f, est.frames, est.dy)
    return _rms_after_gauge(rec_dx, inj_dx), _rms_after_gauge(rec_dy, inj_dy)


def test_recovers_dense_field():
    rx, ry = _recover(structured=False)
    assert rx < 15 and ry < 15, (rx, ry)


def test_recovers_structured_field():
    # The case that broke the naive all-pairs version (spurious far peaks).
    rx, ry = _recover(structured=True)
    assert rx < 15 and ry < 15, (rx, ry)


def test_correction_reduces_drift_spread():
    n_frames = 1500
    f = np.arange(n_frames)
    inj_dx = 200 * f / n_frames
    inj_dy = np.zeros(n_frames)
    locs = _make_locs(1, 3000, n_frames, inj_dx, inj_dy, structured=True)
    est = estimate_drift_rcc(locs, units="nm",
                             params={"n_time_bins": 20, "render_nm": 20})
    corrected = apply_drift(locs, est)
    # Trajectory must be finite and span the injected drift order of magnitude.
    assert np.isfinite(est.dx).all() and np.isfinite(est.dy).all()
    assert np.ptp(est.dx) > 100  # recovered ~200 nm linear ramp
    assert len(corrected) == len(locs)


if __name__ == "__main__":
    print("dense     :", _recover(False))
    print("structured:", _recover(True))
    test_recovers_dense_field()
    test_recovers_structured_field()
    test_correction_reduces_drift_spread()
    print("all tests passed")
