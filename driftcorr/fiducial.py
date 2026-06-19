"""
driftcorr.fiducial

Fiducial-marker drift correction -- the classical, most direct method when bright
persistent markers (gold beads / fluorescent fiducials) are in the field of view.

Unlike a blinking single molecule (present in only a few frames), a fiducial appears
in (almost) every frame, so its apparent motion over time *is* the sample drift. We
detect the fiducials, track each one frame to frame, and average their displacement
into a per-frame drift trajectory. Pure numpy, so it runs in the core env like
`none` / `rcc` (no extra dependencies).

    estimate_drift_fiducial(locs, *, pixel_size_nm, units, params) -> DriftEstimate

params:
    fiducials          optional [[x, y], ...] reference positions (skips auto-detect)
    search_radius_nm   grid cell for auto-detection; choose > the expected total
                       drift so a marker stays in one cell (default 100)
    min_frame_fraction a fiducial must appear in >= this fraction of frames (0.5)
    link_radius_nm     max frame-to-frame jump when tracking (default = search_radius_nm)
    smooth_window      odd moving-average window over the trajectory (default 1 = off)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .core import DriftEstimate


def _detect(x, y, frame, radius, min_frac, n_frames):
    """Grid-occupancy detection: a fiducial cell holds localizations in >= min_frac
    of all frames (persistent), unlike sparsely-blinking single molecules."""
    gx = np.floor(x / radius).astype(np.int64)
    gy = np.floor(y / radius).astype(np.int64)
    fr = frame.astype(np.int64)
    cell_frame = np.unique(np.stack([gx, gy, fr], axis=1), axis=0)   # distinct (cell, frame)
    cells, counts = np.unique(cell_frame[:, :2], axis=0, return_counts=True)
    thresh = max(2, int(round(min_frac * n_frames)))
    seeds = []
    for cx, cy in cells[counts >= thresh]:
        m = (gx == cx) & (gy == cy)
        seeds.append([float(x[m].mean()), float(y[m].mean())])
    return np.asarray(seeds, float) if seeds else np.empty((0, 2), float)


def _track(seed, uframes, by_frame, link2, has_z):
    """Nearest-neighbour link of one fiducial across frames (follows the drift)."""
    cur_x, cur_y = float(seed[0]), float(seed[1])
    rf, rx, ry, rz = [], [], [], []
    for fr in uframes:
        g = by_frame.get(int(fr))
        if g is None:
            continue
        xs, ys, zs = g
        d2 = (xs - cur_x) ** 2 + (ys - cur_y) ** 2
        j = int(np.argmin(d2))
        if d2[j] <= link2:
            cur_x, cur_y = float(xs[j]), float(ys[j])
            rf.append(int(fr)); rx.append(cur_x); ry.append(cur_y)
            if has_z:
                rz.append(float(zs[j]))
    return (np.asarray(rf, float), np.asarray(rx, float), np.asarray(ry, float),
            np.asarray(rz, float) if has_z else None)


def _smooth(a, w):
    w = int(w) | 1                                   # force odd window
    return np.convolve(a, np.ones(w) / w, mode="same")


def estimate_drift_fiducial(
    locs,
    *,
    pixel_size_nm: Optional[float] = None,
    units: str = "nm",
    params: Optional[Dict[str, Any]] = None,
) -> DriftEstimate:
    p = dict(params or {})
    radius = float(p.get("search_radius_nm", 100.0))
    min_frac = float(p.get("min_frame_fraction", 0.5))
    link = float(p.get("link_radius_nm", radius))
    smooth = int(p.get("smooth_window", 1))
    explicit = p.get("fiducials")

    frame = locs["frame"].to_numpy()
    x = locs["x"].to_numpy(float)
    y = locs["y"].to_numpy(float)
    has_z = "z" in locs.columns and np.isfinite(locs["z"].to_numpy(float)).any()
    z = locs["z"].to_numpy(float) if has_z else np.zeros(len(locs))

    uframes = np.unique(frame).astype(np.int64)
    F = uframes.size
    if F < 2:
        return DriftEstimate.zero(locs, units=units, method="fiducial")

    seeds = (np.asarray(explicit, float)[:, :2] if explicit
             else _detect(x, y, frame, radius, min_frac, F))
    if len(seeds) == 0:
        raise ValueError(
            "fiducial drift: no fiducial markers detected (need bright emitters present "
            "in >= min_frame_fraction of frames). Lower min_frame_fraction, raise "
            "search_radius_nm, or pass fiducials=[[x,y], ...].")

    # group localizations by frame once (sorted -> contiguous slices)
    fr_int = frame.astype(np.int64)
    order = np.argsort(fr_int, kind="stable")
    fr_s, x_s, y_s, z_s = fr_int[order], x[order], y[order], z[order]
    starts = np.searchsorted(fr_s, uframes, side="left")
    ends = np.r_[starts[1:], len(fr_s)]
    by_frame = {int(fr): (x_s[a:b], y_s[a:b], z_s[a:b])
                for fr, a, b in zip(uframes, starts, ends)}

    link2 = link * link
    min_track = max(2, int(round(0.25 * min_frac * F)))
    dxs, dys, dzs = [], [], []
    for seed in seeds:
        rf, rx, ry, rz = _track(seed, uframes, by_frame, link2, has_z)
        if rf.size < min_track:
            continue                                 # poorly tracked -> drop
        # displacement vs this fiducial's own mean, interpolated onto every frame
        dxs.append(np.interp(uframes, rf, rx - rx.mean()))
        dys.append(np.interp(uframes, rf, ry - ry.mean()))
        if has_z:
            dzs.append(np.interp(uframes, rf, rz - rz.mean()))

    if not dxs:
        raise ValueError("fiducial drift: candidate markers found but none tracked through "
                         "enough frames (raise link_radius_nm / lower min_frame_fraction).")

    dx = np.mean(dxs, axis=0)
    dy = np.mean(dys, axis=0)
    dz = np.mean(dzs, axis=0) if (has_z and dzs) else np.zeros(F)

    if smooth > 1:
        dx, dy, dz = _smooth(dx, smooth), _smooth(dy, smooth), _smooth(dz, smooth)

    # anchor to the first frame: corrected positions align to the start of the run
    dx, dy, dz = dx - dx[0], dy - dy[0], dz - dz[0]

    return DriftEstimate(
        frames=uframes.astype(float), dx=dx, dy=dy, dz=dz,
        method="fiducial", units=units,
        params={"search_radius_nm": radius, "min_frame_fraction": min_frac,
                "link_radius_nm": link, "smooth_window": smooth, "explicit": bool(explicit)},
        extra={"n_fiducials": int(len(dxs))},
    )
