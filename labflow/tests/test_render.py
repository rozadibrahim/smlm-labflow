"""
SR rendering writes a TIFF that conserves the localization count (histogram mode).

Run: python -m pytest labflow/tests/test_render.py
(or:  python labflow/tests/test_render.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.stages.render import run


def test_render_histogram_writes_image():
    import tifffile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "locs.csv")
        pd.DataFrame({"frame": range(500),
                      "x": rng.uniform(0, 1000, 500),
                      "y": rng.uniform(0, 1000, 500)}).to_csv(inp, index=False)
        out = os.path.join(tmp, "sr.tif")
        run(input_csv=inp, output_csv=out, params={"render_nm": 10, "mode": "histogram"})

        assert os.path.exists(out)
        img = tifffile.imread(out)
        assert img.ndim == 2 and min(img.shape) > 50
        assert abs(img.sum() - 500) < 1e-3, "histogram render should conserve localization count"


if __name__ == "__main__":
    test_render_histogram_writes_image()
    print("render test passed")
