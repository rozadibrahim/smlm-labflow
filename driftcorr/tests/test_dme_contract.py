"""Check the upstream API boundary without needing DME's compiled library."""
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from driftcorr.dme import estimate_drift_dme


@pytest.mark.parametrize("units,scale", [("nm", 100.0), ("pixel", 1.0)])
def test_positions_and_precision_are_pixels_not_variances(monkeypatch, units, scale):
    module = ModuleType("dme.dme")
    def upstream(positions, frames, precision, **kwargs):
        np.testing.assert_allclose(positions, [[1, 2, 3], [2, 3, 4]])
        np.testing.assert_array_equal(frames, [0, 1])
        np.testing.assert_allclose(precision, [[0.1]*3, [0.2]*3])
        assert kwargs["display"] is False
        return np.array([[0, 0, 0], [0.5, -0.5, 1]]), None
    module.dme_estimate = upstream
    monkeypatch.setitem(sys.modules, "dme", ModuleType("dme"))
    monkeypatch.setitem(sys.modules, "dme.dme", module)
    locs = pd.DataFrame({"frame": [5, 6], "x": np.array([1, 2])*scale,
                         "y": np.array([2, 3])*scale, "z": np.array([3, 4])*scale,
                         "lpx": np.array([0.1, 0.2])*scale})
    estimate = estimate_drift_dme(locs, pixel_size_nm=100, units=units)
    np.testing.assert_allclose(estimate.dx, [0, 50])
    np.testing.assert_allclose(estimate.dy, [0, -50])
    np.testing.assert_allclose(estimate.dz, [0, 100])
