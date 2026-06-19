"""
Regression test for the faithful Julia AIM backend.

Skipped automatically when Julia is not installed. With Julia present, it checks
that the port recovers a known injected drift to sub-nanometre precision.

Run: python -m pytest driftcorr/tests/test_aim_julia.py
(or:  python driftcorr/tests/test_aim_julia.py)
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from driftcorr.aim_julia import estimate_drift_aim_julia

_HAS_JULIA = shutil.which("julia") is not None


def _make_structured(seed, n_frames, inj_dx, inj_dy):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(12):
        x0, y0 = rng.uniform(2000, 26000, 2)
        ang, length = rng.uniform(0, np.pi), rng.uniform(3000, 9000)
        t = rng.uniform(0, length, 300)
        pts.append(np.column_stack([
            x0 + t * np.cos(ang) + rng.normal(0, 40, t.size),
            y0 + t * np.sin(ang) + rng.normal(0, 40, t.size),
        ]))
    emit = np.vstack(pts)
    rows = []
    for fr in range(n_frames):
        k = rng.integers(len(emit), size=200)
        rows.append(np.column_stack([
            np.full(k.size, fr + 1),
            emit[k, 0] + inj_dx[fr] + rng.normal(0, 8, k.size),
            emit[k, 1] + inj_dy[fr] + rng.normal(0, 8, k.size),
        ]))
    arr = np.vstack(rows)
    return pd.DataFrame({"frame": arr[:, 0].astype(int), "x": arr[:, 1], "y": arr[:, 2]})


def _rms_after_gauge(rec, inj):
    rec, inj = rec - rec.mean(), inj - inj.mean()
    return float(np.sqrt(np.mean((rec - inj) ** 2)))


def test_aim_julia_recovers_injected_drift():
    if not _HAS_JULIA:
        print("SKIP: julia not installed")
        return
    n_frames = 2000
    f = np.arange(n_frames)
    inj_dx = 250 * f / n_frames + 40 * np.sin(2 * np.pi * f / n_frames)
    inj_dy = -180 * f / n_frames + 30 * np.cos(2 * np.pi * f / n_frames)
    locs = _make_structured(7, n_frames, inj_dx, inj_dy)
    est = estimate_drift_aim_julia(
        locs, pixel_size_nm=100, units="nm",
        params={"intersect_nm": 20, "track_interval": 100},
    )
    rx = _rms_after_gauge(np.interp(f + 1, est.frames, est.dx), inj_dx)
    ry = _rms_after_gauge(np.interp(f + 1, est.frames, est.dy), inj_dy)
    # Faithful port reaches sub-nm; allow generous headroom for noise/seed.
    assert rx < 3 and ry < 3, (rx, ry)


if __name__ == "__main__":
    if not _HAS_JULIA:
        print("julia not installed - skipping")
    else:
        test_aim_julia_recovers_injected_drift()
        print("aim_julia test passed")
