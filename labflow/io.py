"""
labflow.io

Canonical localization IO -- the file-contract layer of the spine.

Every stage reads and writes the canonical localization CSV (see schema.py).
Keeping that read/write in one place means a new method, in any stage, plumbs its
files through a single import instead of re-implementing CSV handling (or
borrowing it from another backend package):

    from labflow.io import read_localizations, read_table, write_table

  read_localizations  alias-aware canonical loader: returns a frame/x/y[/z/lpx/lpy]
                      table and also accepts raw ThunderSTORM / Picasso headers
                      ("x [nm]", "frame_id", ...) with no separate pass. The full
                      original frame is kept on `df.attrs["raw"]` so a stage can
                      carry extra columns (photons, etc.) through to its output.
  read_table          plain passthrough read (pd.read_csv) for stages that consume
                      a non-localization contract (clusters.csv, tracks.csv, ...).
  write_table         write a DataFrame to a CSV, creating parent dirs, and return
                      the path -- the single line every in-process adapter ends on.

This module imports only numpy/pandas, so it never pulls a backend into the core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Canonical column aliases so methods also accept raw ThunderSTORM / Picasso
# headers without a separate canonicalization pass.
_ALIASES: Dict[str, List[str]] = {
    "frame": ["frame", "Frame", "frame_id", "t"],
    "x": ["x", "x [nm]", "x_nm", "x[nm]", "xnm"],
    "y": ["y", "y [nm]", "y_nm", "y[nm]", "ynm"],
    "z": ["z", "z [nm]", "z_nm", "z[nm]", "znm"],
    "lpx": ["lpx", "uncertainty_xy [nm]", "uncertainty [nm]", "lpx_nm"],
    "lpy": ["lpy", "uncertainty_xy [nm]", "uncertainty [nm]", "lpy_nm"],
}


def _pick(df: pd.DataFrame, canonical: str) -> Optional[str]:
    for name in _ALIASES.get(canonical, [canonical]):
        if name in df.columns:
            return name
    return None


def read_localizations(path: Union[str, Path]) -> pd.DataFrame:
    """Load a localization CSV into a frame/x/y[/z/lpx/lpy] table.

    Required columns (after alias resolution): frame, x, y. The untouched source
    frame is preserved on `df.attrs["raw"]` so downstream output can carry the
    original extra columns (photons, background, ...).
    """
    raw = pd.read_csv(path)
    out: Dict[str, np.ndarray] = {}
    for canonical in ("frame", "x", "y", "z", "lpx", "lpy"):
        col = _pick(raw, canonical)
        if col is not None:
            out[canonical] = pd.to_numeric(raw[col], errors="coerce").to_numpy()

    missing = [c for c in ("frame", "x", "y") if c not in out]
    if missing:
        raise ValueError(
            f"Localization file {path} is missing required column(s): {missing}. "
            f"Found columns: {list(raw.columns)}"
        )

    df = pd.DataFrame(out)
    df = df.dropna(subset=["frame", "x", "y"]).reset_index(drop=True)
    df["frame"] = df["frame"].astype(np.int64)
    df.attrs["raw"] = raw          # keep originals so output can pass them through
    return df


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Plain CSV read for a non-localization contract (clusters/tracks/etc.)."""
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Union[str, Path]) -> str:
    """Write `df` to `path` (creating parent dirs); return the path as a str.

    The canonical end of an in-process adapter:  return write_table(out, output_csv)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p)
