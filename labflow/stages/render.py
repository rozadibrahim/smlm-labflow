"""
labflow.stages.render

Super-resolution image rendering: localizations -> a reconstructed SR image (the
visual output, and the input some segmenters consume). Modes: `histogram` (a 2D
count image) or `gaussian` (each localization blurred by a kernel). numpy (+ scipy
for the gaussian mode); writes a float32 TIFF.

Contract: localizations CSV in -> sr_image.tif out (a rendered image, not a table;
so this stage has no downstream CSV contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..io import read_localizations


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    import tifffile

    p = dict(params or {})
    p.pop("pixel_size_nm", None)
    p.pop("units", None)
    render_nm = float(p.get("render_nm", 10.0))
    mode = str(p.get("mode", "histogram")).lower()

    locs = read_localizations(input_csv)
    x = locs["x"].to_numpy(float)
    y = locs["y"].to_numpy(float)
    xmin, ymin = float(x.min()), float(y.min())
    nx = max(int((x.max() - xmin) / render_nm) + 1, 1)
    ny = max(int((y.max() - ymin) / render_nm) + 1, 1)
    img, _, _ = np.histogram2d(y, x, bins=[ny, nx],
                               range=[[ymin, float(y.max())], [xmin, float(x.max())]])

    if mode == "gaussian":
        from scipy.ndimage import gaussian_filter
        sigma_px = max(float(p.get("sigma_nm", render_nm)) / render_nm, 0.1)
        img = gaussian_filter(img, sigma_px)

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out, img.astype(np.float32))
    print(f"render ({mode}): {len(x):,} localizations -> {ny}x{nx} image, {out}")
    return str(out)
