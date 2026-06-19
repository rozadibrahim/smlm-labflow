"""
labflow.stages.analyze_msd

Per-track mean-squared-displacement (MSD) diffusion analysis -- the classical SPT
readout. For each trajectory it fits:
  - the diffusion coefficient D (2D: MSD(t) = 4 D t + offset; the offset absorbs
    localization error, so the slope gives D);
  - the anomalous exponent alpha (MSD ~ t^alpha; <1 sub-diffusive/confined,
    ~1 Brownian, >1 super-diffusive/directed) from a log-log fit;
  - track length, duration, and radius of gyration.
Pure numpy/pandas -> runs in the core env.

Contract: tracks.csv (track_id, frame, x, y) in -> track_analysis.csv out (one row per
track; schema.ANALYSIS_COLUMNS).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..io import read_table, write_table


def _metrics(g: pd.DataFrame, max_lag: int, dt: float, min_len: int) -> Optional[dict]:
    g = g.sort_values("frame")
    xy = g[["x", "y"]].to_numpy(float)
    frames = g["frame"].to_numpy(float)
    n = len(xy)
    if n < min_len:
        return None

    n_lag = int(min(max_lag, n - 1))
    lags = np.arange(1, n_lag + 1)
    msd = np.array([np.mean(np.sum((xy[k:] - xy[:-k]) ** 2, axis=1)) for k in lags])
    t = lags * dt

    slope = np.polyfit(t, msd, 1)[0] if n_lag >= 2 else msd[0] / t[0]
    D = max(slope / 4.0, 0.0)                       # 2D Brownian: MSD = 4 D t

    good = msd > 0
    alpha = (np.polyfit(np.log(t[good]), np.log(msd[good]), 1)[0]
             if good.sum() >= 2 else np.nan)

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    centroid = xy.mean(axis=0)
    return {
        "track_id": int(g["track_id"].iloc[0]),
        "n_localizations": int(n),
        "length_nm": float(steps.sum()),
        "duration_frames": float(frames.max() - frames.min() + 1),
        "diffusion_coefficient": float(D),
        "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "radius_gyration_nm": float(np.sqrt(((xy - centroid) ** 2).sum(axis=1).mean())),
    }


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    p.pop("pixel_size_nm", None)
    p.pop("units", None)
    max_lag = int(p.get("max_lag", 4))
    dt = float(p.get("frame_interval", 1.0))
    min_len = int(p.get("min_length", 3))

    df = read_table(input_csv)
    if "track_id" not in df.columns:
        raise ValueError("analyze needs a 'track_id' column - run the track stage first.")

    rows = []
    for _, g in df.groupby("track_id"):
        m = _metrics(g, max_lag, dt, min_len)
        if m is not None:
            rows.append(m)
    out = pd.DataFrame(rows)

    n_tracks = len(out)
    d_med = float(out["diffusion_coefficient"].median()) if n_tracks else float("nan")
    print(f"msd: {n_tracks} tracks analysed, median D={d_med:.1f} nm^2/frame")
    return write_table(out, output_csv)
