"""
driftcorr.dme

Drift at Minimum Entropy (DME) backend, after Cnossen, Cui, Joo & Smith,
"Drift correction in localization microscopy using entropy minimization",
Opt. Express 29, 27961 (2021). State-of-the-art marker-free drift estimation
(~5x better precision than RCC at comparable compute, 2D/3D).

This is a thin adapter over the upstream package
(https://github.com/qnano/drift-estimation): we map canonical localizations to
the `dme_estimate` calling convention and translate its per-frame drift (in
pixels) back into a nm trajectory. The heavy lifting stays upstream.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .core import DriftEstimate


def estimate_drift_dme(
    locs: pd.DataFrame,
    *,
    pixel_size_nm: Optional[float] = None,
    units: str = "nm",
    params: Optional[Dict[str, Any]] = None,
) -> DriftEstimate:
    try:
        from dme.dme import dme_estimate
    except Exception as exc:  # pragma: no cover - exercised only without dme
        raise RuntimeError(
            "The 'dme' backend needs the upstream DME package "
            "(https://github.com/qnano/drift-estimation). Install it and its "
            "build/CUDA dependencies, or select drift_backend: rcc."
        ) from exc

    params = dict(params or {})
    if not pixel_size_nm:
        raise ValueError("DME requires --pixel-size (positions are computed in pixels).")

    x = locs["x"].to_numpy(float)
    y = locs["y"].to_numpy(float)
    frame = locs["frame"].to_numpy(np.int64)

    # DME works in pixel units.
    if units == "nm":
        px, py = x / pixel_size_nm, y / pixel_size_nm
    else:
        px, py = x, y

    has_z = "z" in locs.columns and np.isfinite(locs["z"].to_numpy(float)).any()
    if has_z:
        pz = locs["z"].to_numpy(float) / pixel_size_nm
        positions = np.stack([px, py, pz], axis=1)
    else:
        positions = np.stack([px, py], axis=1)

    framenum = (frame - frame.min()).astype(np.int32)
    n_frames = int(framenum.max()) + 1
    ndim = positions.shape[1]

    # CRLB (localization precision squared, in px^2) is used as per-spot weight.
    default_prec_px = float(params.get("default_precision_px", 0.2))
    if "lpx" in locs.columns:
        prec = np.nan_to_num(
            locs["lpx"].to_numpy(float) / pixel_size_nm, nan=default_prec_px
        )
        prec = np.where(prec > 0, prec, default_prec_px)
    else:
        prec = np.full(positions.shape[0], default_prec_px)
    crlb = np.tile((prec ** 2)[:, None], (1, ndim))

    fov = int(np.ceil(max(px.max(), py.max()))) + 1
    coarse_sigma = params.get("coarse_sigma", [0.2] * ndim)

    estimated_drift, _ = dme_estimate(
        positions,
        framenum,
        crlb,
        framesperbin=int(params.get("frames_per_bin", 1)),
        imgshape=[fov, fov],
        coarseFramesPerBin=int(params.get("coarse_frames_per_bin", 200)),
        coarseSigma=coarse_sigma,
        useCuda=bool(params.get("use_cuda", False)),
        useDebugLibrary=False,
    )

    estimated_drift = np.asarray(estimated_drift, dtype=float)  # (n_frames, ndim), px
    drift_nm = estimated_drift * pixel_size_nm

    uniq = np.unique(frame)
    rel = (uniq - frame.min()).astype(int)
    rel = np.clip(rel, 0, drift_nm.shape[0] - 1)
    dx = drift_nm[rel, 0]
    dy = drift_nm[rel, 1]
    dz = drift_nm[rel, 2] if ndim == 3 else np.zeros_like(dx)

    return DriftEstimate(
        frames=uniq, dx=dx, dy=dy, dz=dz, method="dme", units="nm",
        params={k: params[k] for k in params},
        extra={"ndim": ndim, "n_frames": n_frames},
    )
