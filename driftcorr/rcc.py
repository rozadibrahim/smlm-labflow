"""
driftcorr.rcc

Redundant cross-correlation (RCC) drift estimation, after
Wang et al., "Localization events-based sample drift correction for
localization microscopy with redundant cross-correlation algorithm",
Opt. Express 22, 15982 (2014).

Pure numpy: localizations are split into time blocks, each rendered to a 2D
histogram, all pairwise block shifts are measured by FFT cross-correlation with
sub-pixel parabolic refinement, and a redundant least-squares fit turns the
over-determined set of pairwise shifts into a per-block drift curve, which is
then interpolated to per-frame.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .core import DriftEstimate


def _render(x: np.ndarray, y: np.ndarray, x0: float, y0: float,
            nx: int, ny: int, bin_nm: float) -> np.ndarray:
    ix = np.clip(((x - x0) / bin_nm).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / bin_nm).astype(np.int64), 0, ny - 1)
    img = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(img, (iy, ix), 1.0)
    # Light smoothing stabilises the correlation peak for sparse blocks.
    img -= img.mean()
    return img


def _parabolic(c: np.ndarray, peak: int) -> float:
    """Sub-sample peak offset from a 1D 3-point parabola around `peak`."""
    if peak <= 0 or peak >= c.size - 1:
        return 0.0
    a, b, d = c[peak - 1], c[peak], c[peak + 1]
    denom = (a - 2.0 * b + d)
    if denom == 0.0:
        return 0.0
    return 0.5 * (a - d) / denom


def _xcorr_shift(ref: np.ndarray, mov: np.ndarray,
                 max_shift_px: Optional[float] = None) -> tuple[float, float]:
    """Shift (dy, dx) such that mov(pos) ~= ref(pos - shift).

    `max_shift_px` restricts the peak search to a window around zero shift,
    which rejects spurious far-field correlation peaks from repetitive sample
    structure (drift between nearby time blocks is always small).
    """
    f_ref = np.fft.rfft2(ref)
    f_mov = np.fft.rfft2(mov)
    cc = np.fft.irfft2(f_ref * np.conj(f_mov), s=ref.shape)
    cc = np.fft.fftshift(cc)
    cy, cx = ref.shape[0] // 2, ref.shape[1] // 2

    if max_shift_px is not None and max_shift_px > 0:
        # Locate the peak inside the allowed window, but refine on the original
        # correlation so the parabolic fit never reads masked-out neighbours.
        m = int(np.ceil(max_shift_px))
        y0, y1 = max(0, cy - m), min(cc.shape[0], cy + m + 1)
        x0, x1 = max(0, cx - m), min(cc.shape[1], cx + m + 1)
        sub = cc[y0:y1, x0:x1]
        iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
        py, px = y0 + iy, x0 + ix
    else:
        py, px = np.unravel_index(int(np.argmax(cc)), cc.shape)

    sy = (py - cy) + _parabolic(cc[:, px], py)
    sx = (px - cx) + _parabolic(cc[py, :], px)
    # Peak is the displacement of `ref` w.r.t. `mov`; negate so the result is
    # the displacement of `mov` relative to `ref` (mov(p) ~= ref(p - shift)).
    return -float(sy), -float(sx)


def estimate_drift_rcc(
    locs: pd.DataFrame,
    *,
    pixel_size_nm: Optional[float] = None,
    units: str = "nm",
    params: Optional[Dict[str, Any]] = None,
) -> DriftEstimate:
    params = dict(params or {})
    n_blocks = int(params.get("n_time_bins", 25))
    bin_nm = float(params.get("render_nm", 20.0))
    max_grid = int(params.get("max_grid", 512))
    # Robustness controls (see module docstring).
    neighbor_span = int(params.get("neighbor_span", 6))
    max_drift_nm = float(params.get("max_drift_nm", 1000.0))

    x = locs["x"].to_numpy(float)
    y = locs["y"].to_numpy(float)
    frame = locs["frame"].to_numpy(np.int64)

    if units == "pixel" and pixel_size_nm:
        x = x * pixel_size_nm
        y = y * pixel_size_nm

    uniq_frames = np.unique(frame)
    n_blocks = max(2, min(n_blocks, uniq_frames.size))

    # Common render grid for every block.
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    nx = int(np.ceil((x1 - x0) / bin_nm)) + 1
    ny = int(np.ceil((y1 - y0) / bin_nm)) + 1
    if max(nx, ny) > max_grid:
        bin_nm *= max(nx, ny) / max_grid
        nx = int(np.ceil((x1 - x0) / bin_nm)) + 1
        ny = int(np.ceil((y1 - y0) / bin_nm)) + 1

    # Assign each localization to a time block by frame range.
    edges = np.linspace(uniq_frames.min(), uniq_frames.max() + 1, n_blocks + 1)
    block = np.clip(np.digitize(frame, edges) - 1, 0, n_blocks - 1)

    images = []
    block_frame = []
    for b in range(n_blocks):
        m = block == b
        if m.sum() < 10:
            images.append(None)
            block_frame.append(np.nan)
            continue
        images.append(_render(x[m], y[m], x0, y0, nx, ny, bin_nm))
        block_frame.append(float(np.median(frame[m])))

    valid = [b for b in range(n_blocks) if images[b] is not None]
    if len(valid) < 2:
        return DriftEstimate.zero(locs, units="nm", method="rcc")

    # Redundant cross-correlation: measure relative shift only between nearby
    # block pairs (drift over a few blocks is bounded), each search constrained
    # to a max-drift window so spurious far peaks are ignored.
    max_shift_px = max_drift_nm / bin_nm
    rows, sy_list, sx_list = [], [], []
    for ii in range(len(valid)):
        for jj in range(ii + 1, min(ii + 1 + neighbor_span, len(valid))):
            sy, sx = _xcorr_shift(images[valid[ii]], images[valid[jj]],
                                  max_shift_px=max_shift_px)
            row = np.zeros(len(valid))
            row[ii] = -1.0
            row[jj] = 1.0
            rows.append(row)
            sy_list.append(sy * bin_nm)
            sx_list.append(sx * bin_nm)

    A_full = np.vstack(rows)[:, 1:]   # gauge fix: first valid block = 0
    sy_arr, sx_arr = np.array(sy_list), np.array(sx_list)

    def _solve_irls(A: np.ndarray, rhs: np.ndarray, iters: int = 3) -> np.ndarray:
        w = np.ones(rhs.shape[0])
        sol = np.zeros(A.shape[1])
        for _ in range(iters):
            Aw, bw = A * w[:, None], rhs * w
            sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            resid = np.abs(A @ sol - rhs)
            mad = np.median(resid) + 1e-9
            w = (resid <= 4.0 * mad).astype(float)   # drop gross outliers
            if w.sum() < A.shape[1]:                 # keep system solvable
                w = np.ones_like(w)
                break
        return sol

    dy_sol = _solve_irls(A_full, sy_arr)
    dx_sol = _solve_irls(A_full, sx_arr)
    block_dx = np.concatenate([[0.0], dx_sol])
    block_dy = np.concatenate([[0.0], dy_sol])

    bf = np.array([block_frame[b] for b in valid], dtype=float)
    order = np.argsort(bf)
    bf, block_dx, block_dy = bf[order], block_dx[order], block_dy[order]

    # Interpolate per-block drift to every unique frame.
    fx = uniq_frames.astype(float)
    dx = np.interp(fx, bf, block_dx)
    dy = np.interp(fx, bf, block_dy)
    dz = np.zeros_like(dx)

    est = DriftEstimate(
        frames=uniq_frames, dx=dx, dy=dy, dz=dz,
        method="rcc", units="nm",
        params={"n_time_bins": n_blocks, "render_nm": bin_nm, "max_grid": max_grid,
                "neighbor_span": neighbor_span, "max_drift_nm": max_drift_nm},
        extra={"n_blocks_used": len(valid), "grid": [nx, ny]},
    )
    return est
