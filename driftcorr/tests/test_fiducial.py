"""
Fiducial-marker drift correction recovers a known injected drift.

Run: python -m pytest driftcorr/tests/test_fiducial.py
(or:  python driftcorr/tests/test_fiducial.py)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from driftcorr.core import apply_drift
from driftcorr.fiducial import estimate_drift_fiducial


def _synth(n_frames=300, seed=0):
    """Two bright fiducials + sparse blinking background, shifted by a known drift."""
    rng = np.random.default_rng(seed)
    f = np.arange(n_frames)
    drift = np.c_[20.0 * np.sin(2 * np.pi * f / n_frames), 50.0 * f / n_frames]  # known, drift[0]=0
    fids = np.array([[2050.0, 2050.0], [4050.0, 3050.0]])
    rows = []
    for fi in range(n_frames):
        for x, y in rng.uniform(500, 5000, (4, 2)) + rng.normal(0, 5, (4, 2)):   # blinkers
            rows.append({"frame": fi, "x": float(x), "y": float(y)})
        for x, y in fids + drift[fi] + rng.normal(0, 1.0, fids.shape):           # fiducials
            rows.append({"frame": fi, "x": float(x), "y": float(y)})
    return pd.DataFrame(rows), drift


def test_recovers_known_drift():
    locs, drift = _synth()
    est = estimate_drift_fiducial(locs, params={"search_radius_nm": 100, "min_frame_fraction": 0.5})

    assert est.method == "fiducial" and est.extra["n_fiducials"] == 2
    # injected drift is already anchored at frame 0 ([0,0]); compare per-frame.
    assert np.abs(est.dx - drift[:, 0]).max() < 5.0, np.abs(est.dx - drift[:, 0]).max()
    assert np.abs(est.dy - drift[:, 1]).max() < 5.0, np.abs(est.dy - drift[:, 1]).max()


def test_correction_shrinks_fiducial_spread():
    locs, _ = _synth()
    est = estimate_drift_fiducial(locs, params={"search_radius_nm": 100})
    corrected = apply_drift(locs, est)
    # the brightest cluster (a fiducial) should be tighter after correction
    before = locs[(np.hypot(locs.x - 2050, locs.y - 2050) < 80)]
    after = corrected.loc[before.index]
    assert after["x"].std() < before["x"].std()
    assert after["y"].std() < before["y"].std()


def test_no_fiducials_is_clear_error():
    rng = np.random.default_rng(1)
    # only sparse blinkers, no persistent marker -> must fail with a clear message
    locs = pd.DataFrame({"frame": np.repeat(np.arange(100), 3),
                         "x": rng.uniform(0, 5000, 300), "y": rng.uniform(0, 5000, 300)})
    try:
        estimate_drift_fiducial(locs, params={"min_frame_fraction": 0.5})
    except ValueError as exc:
        assert "no fiducial" in str(exc).lower()
    else:
        raise AssertionError("expected a ValueError when no fiducials are present")


if __name__ == "__main__":
    test_recovers_known_drift()
    test_correction_shrinks_fiducial_spread()
    test_no_fiducials_is_clear_error()
    print("fiducial tests passed")
