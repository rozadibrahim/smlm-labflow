"""
adapters._localize_io

Standalone IO shared by the localizer adapters (DECODE / FD-DeepLoc / DeepSTORM3D).

These adapters run INSIDE each tool's own conda env, which does not have labflow
installed -- so this helper imports only numpy / pandas / tifffile and never labflow.
It reads a frame stack and writes the canonical localization CSV (the columns in
schema.py), which the labflow runner then validates against the file contract and
chains downstream. Files are the only thing that crosses the env boundary.
"""

import numpy as np
import pandas as pd

try:
    import tifffile
except ImportError:                       # adapters add tifffile to their env spec
    tifffile = None

# Canonical localization columns (mirrors schema.CANONICAL_COLUMNS).
CANONICAL = ["frame", "x", "y", "z", "photons", "background",
             "confidence", "backend", "source_file"]


def read_frames(path):
    """Load a (multi-page) TIFF stack into an (F, H, W) array."""
    if tifffile is None:
        raise RuntimeError("tifffile is required to read the frame stack "
                           "(add it to the tool's env spec).")
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:                     # single frame -> (1, H, W)
        arr = arr[None]
    return arr


def write_localizations(out_csv, *, frame, x, y, z=None, photons=None,
                        backend="", source_file=""):
    """Write per-localization arrays as a canonical localization CSV."""
    n = len(frame)
    df = pd.DataFrame({
        "frame": np.asarray(frame, dtype=int),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": (np.asarray(z, dtype=float) if z is not None else np.zeros(n)),
        "photons": (np.asarray(photons, dtype=float) if photons is not None
                    else np.full(n, np.nan)),
        "background": np.nan,
        "confidence": np.nan,
        "backend": backend,
        "source_file": str(source_file),
    })
    df.to_csv(out_csv, index=False)
    return out_csv
