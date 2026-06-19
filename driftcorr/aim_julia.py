"""
driftcorr.aim_julia

Backend that runs the *faithful* Julia port of the authors' AIM MATLAB code
(driftcorr/julia/aim.jl) as a subprocess. Unlike the pure-numpy `aim` backend
(a from-paper reimplementation), this is a near-line-by-line translation of
AIM.m / IntersectionMax.m, so it reproduces their algorithm — including the
fixed-reference pre-shift tracking, the FFT-phase sub-pixel peak, the two-round
scheme, and the spline interpolation — and recovers injected drift to sub-nm.

Requires a Julia runtime on PATH (or set `julia_bin`). No Julia packages needed;
the port uses only the Julia standard library.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .core import DriftEstimate

_SCRIPT = Path(__file__).resolve().parent / "julia" / "aim.jl"


def _find_julia(params: Dict[str, Any]) -> str:
    candidate = params.get("julia_bin") or "julia"
    found = shutil.which(candidate)
    if found:
        return found
    raise RuntimeError(
        f"Julia executable {candidate!r} not found on PATH. Install Julia "
        "(https://julialang.org/ or `winget install Julialang.Juliaup`) or set "
        "the `julia_bin` parameter / drift_backend rcc."
    )


def estimate_drift_aim_julia(
    locs: pd.DataFrame,
    *,
    pixel_size_nm: Optional[float] = None,
    units: str = "nm",
    params: Optional[Dict[str, Any]] = None,
) -> DriftEstimate:
    params = dict(params or {})
    intersect_nm = float(params.get("intersect_nm", 20.0))
    track_interval = int(params.get("track_interval", 500))

    julia = _find_julia(params)
    if not _SCRIPT.exists():
        raise RuntimeError(f"AIM Julia script not found: {_SCRIPT}")

    x = locs["x"].to_numpy(float)
    y = locs["y"].to_numpy(float)
    frame = locs["frame"].to_numpy(np.int64)

    if units == "pixel" and pixel_size_nm:           # AIM runs in nm here
        x = x * pixel_size_nm
        y = y * pixel_size_nm

    with tempfile.TemporaryDirectory(prefix="aim_jl_") as tmp:
        in_csv = Path(tmp) / "in.csv"
        out_csv = Path(tmp) / "out.csv"
        pd.DataFrame({"frame": frame, "x": x, "y": y}).to_csv(in_csv, index=False)

        proc = subprocess.run(
            [julia, str(_SCRIPT),
             "--in", str(in_csv), "--out", str(out_csv),
             "--track-interval", str(track_interval),
             "--intersect", str(intersect_nm)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Julia AIM failed (rc="
                f"{proc.returncode}).\nstdout:\n{proc.stdout[-1500:]}\n"
                f"stderr:\n{proc.stderr[-2000:]}"
            )

        traj = pd.read_csv(out_csv)

    frames = traj["frame"].to_numpy(np.int64)
    dx = traj["dx"].to_numpy(float)
    dy = traj["dy"].to_numpy(float)
    dz = np.zeros_like(dx)

    return DriftEstimate(
        frames=frames, dx=dx, dy=dy, dz=dz, method="aim_julia", units="nm",
        params={"intersect_nm": intersect_nm, "track_interval": track_interval},
        extra={"julia": julia, "source": "faithful port of YangLiuLab/AIM"},
    )
