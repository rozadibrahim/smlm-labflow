"""
driftcorr.core

Shared types and IO for drift-correction backends.

The contract every backend converges on is the canonical localization table
(see schema.py) plus a per-frame drift trajectory. Keeping IO here means each
backend only implements the estimation maths, not file plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Canonical localization IO now lives in the spine (labflow.io); re-exported here
# under its historical name so existing callers keep working unchanged
# (driftcorr.run, the tests, and `from driftcorr.core import load_localizations`).
from labflow.io import read_localizations as load_localizations  # noqa: F401


@dataclass
class DriftEstimate:
    """Per-frame drift trajectory in the same units as the localizations."""

    frames: np.ndarray            # (F,) unique, sorted frame indices
    dx: np.ndarray                # (F,) drift in x to subtract from positions
    dy: np.ndarray                # (F,) drift in y
    dz: np.ndarray                # (F,) drift in z (zeros if 2D)
    method: str = "none"
    units: str = "nm"
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def zero(cls, locs: pd.DataFrame, units: str = "nm", method: str = "none") -> "DriftEstimate":
        frames = np.unique(locs["frame"].to_numpy())
        z = np.zeros(frames.shape[0], dtype=float)
        return cls(frames=frames, dx=z.copy(), dy=z.copy(), dz=z.copy(),
                   method=method, units=units)

    def trajectory_frame(self) -> pd.DataFrame:
        radial = np.sqrt(self.dx ** 2 + self.dy ** 2)
        return pd.DataFrame(
            {
                "frame": self.frames,
                "dx": self.dx,
                "dy": self.dy,
                "dz": self.dz,
                "radial_drift": radial,
                "method": self.method,
                "units": self.units,
            }
        )

    def residual_metrics(self) -> Dict[str, Optional[float]]:
        radial = np.sqrt(self.dx ** 2 + self.dy ** 2)
        if radial.size == 0:
            return {"max_radial_drift": None, "median_radial_drift": None,
                    "p95_radial_drift": None}
        slope = None
        if self.frames.size >= 2 and np.ptp(self.frames) > 0:
            slope = float(np.polyfit(self.frames.astype(float), radial, 1)[0])
        return {
            "max_radial_drift": float(np.max(radial)),
            "median_radial_drift": float(np.median(radial)),
            "p95_radial_drift": float(np.percentile(radial, 95)),
            "linear_radial_drift_slope_per_frame": slope,
        }


def apply_drift(locs: pd.DataFrame, estimate: DriftEstimate) -> pd.DataFrame:
    """Subtract the (interpolated) drift trajectory from each localization."""
    frame = locs["frame"].to_numpy(dtype=float)
    traj_f = estimate.frames.astype(float)

    def interp(values: np.ndarray) -> np.ndarray:
        if traj_f.size == 0:
            return np.zeros_like(frame)
        return np.interp(frame, traj_f, values)

    corrected = locs.copy()
    corrected["x"] = locs["x"].to_numpy(float) - interp(estimate.dx)
    corrected["y"] = locs["y"].to_numpy(float) - interp(estimate.dy)
    if "z" in corrected.columns:
        corrected["z"] = locs["z"].to_numpy(float) - interp(estimate.dz)
    return corrected


def write_outputs(
    out_dir: str | Path,
    *,
    backend: str,
    locs: pd.DataFrame,
    corrected: pd.DataFrame,
    estimate: DriftEstimate,
    source_csv: str,
    pixel_size_nm: Optional[float],
    units: str,
    elapsed_sec: Optional[float] = None,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corrected_csv = out_dir / "drift_corrected_localizations.csv"
    trajectory_csv = out_dir / "drift_trajectory.csv"
    summary_json = out_dir / "drift_correction.json"

    # Write corrected localizations, carrying original extra columns through.
    raw = locs.attrs.get("raw")
    if isinstance(raw, pd.DataFrame) and len(raw) == len(corrected):
        out = raw.copy()
        out["x_corrected"] = corrected["x"].to_numpy()
        out["y_corrected"] = corrected["y"].to_numpy()
        if "z" in corrected.columns:
            out["z_corrected"] = corrected["z"].to_numpy()
        out.to_csv(corrected_csv, index=False)
    else:
        corrected.to_csv(corrected_csv, index=False)

    estimate.trajectory_frame().to_csv(trajectory_csv, index=False)

    summary = {
        "benchmark_layer": "drift_correction",
        "backend": backend,
        "method": estimate.method,
        "status": "passed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(source_csv),
        "units": units,
        "pixel_size_nm": pixel_size_nm,
        "n_localizations": int(len(locs)),
        "n_frames": int(estimate.frames.size),
        "elapsed_sec": elapsed_sec,
        "params": estimate.params,
        **estimate.residual_metrics(),
        **{f"extra_{k}": v for k, v in estimate.extra.items()},
        "corrected_localizations_csv": str(corrected_csv),
        "drift_trajectory_csv": str(trajectory_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "corrected_localizations_csv": str(corrected_csv),
        "drift_trajectory_csv": str(trajectory_csv),
        "drift_correction_json": str(summary_json),
    }
