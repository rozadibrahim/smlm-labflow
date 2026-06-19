"""
labflow.stages.metrics

Resolution and precision quality metrics on a localization table -- the two numbers
every SMLM dataset reports. One adapter; `metric` (bind) selects:

  frc   Fourier Ring Correlation (Nieuwenhuizen et al., Nat. Methods 2013): split the
        localizations into two halves, render each, and correlate their FFTs over rings
        of spatial frequency. The resolution is 1/q where FRC(q) crosses the fixed 1/7
        threshold. Pure numpy.
  nena  Nearest-neighbour based analysis (Endesfelder et al. 2014): the localization
        precision from the distribution of distances between the same molecule
        re-localized in consecutive frames (a Rayleigh fit). numpy + scipy.

Contract: localizations CSV in -> metrics.csv out (metric, value); frc also writes
frc_curve.csv alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..io import read_localizations, write_table


def _render_hist(x, y, render_nm, extent):
    xmin, xmax, ymin, ymax = extent
    nx = max(int((xmax - xmin) / render_nm), 1)
    ny = max(int((ymax - ymin) / render_nm), 1)
    H, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
    return H


def _frc(x, y, render_nm, seed=0):
    """Return (spatial_freq, frc_curve, resolution_nm)."""
    rng = np.random.default_rng(seed)
    half = rng.random(len(x)) < 0.5
    extent = (x.min(), x.max(), y.min(), y.max())
    H1 = _render_hist(x[half], y[half], render_nm, extent)
    H2 = _render_hist(x[~half], y[~half], render_nm, extent)

    F1 = np.fft.fftshift(np.fft.fft2(H1))
    F2 = np.fft.fftshift(np.fft.fft2(H2))
    ny, nx = H1.shape
    Y, X = np.ogrid[:ny, :nx]
    R = np.hypot(X - nx // 2, Y - ny // 2).astype(int)

    n_rings = R.max() + 1
    num = np.zeros(n_rings)
    d1 = np.zeros(n_rings)
    d2 = np.zeros(n_rings)
    np.add.at(num, R.ravel(), (F1 * np.conj(F2)).real.ravel())
    np.add.at(d1, R.ravel(), (np.abs(F1) ** 2).ravel())
    np.add.at(d2, R.ravel(), (np.abs(F2) ** 2).ravel())
    frc = num / np.sqrt(d1 * d2 + 1e-12)

    N = max(nx, ny)
    freq = np.arange(n_rings) / (N * render_nm)        # cycles / nm
    below = np.where(frc < 1.0 / 7.0)[0]
    resolution = (1.0 / freq[below[0]] if len(below) and freq[below[0]] > 0 else np.nan)
    return freq, frc, resolution


def _nena(x, y, frame):
    """Localization precision (nm) from consecutive-frame nearest-neighbour distances."""
    from scipy.optimize import curve_fit
    from scipy.spatial import cKDTree

    frames = frame.astype(int)
    by_frame = {int(f): np.c_[x[frames == f], y[frames == f]] for f in np.unique(frames)}
    dists = []
    for f, cur in by_frame.items():
        nxt = by_frame.get(f + 1)
        if nxt is not None and len(nxt) and len(cur):
            dists.append(cKDTree(nxt).query(cur, k=1)[0])
    d = np.concatenate(dists) if dists else np.array([])
    if d.size < 20:
        raise ValueError("nena needs localizations that reappear in consecutive frames.")

    max_d = float(np.percentile(d, 95))
    d = d[d <= max_d]
    hist, edges = np.histogram(d, bins=50, range=(0.0, max_d))
    centers = 0.5 * (edges[:-1] + edges[1:])

    def model(r, A, sigma, B):                         # Rayleigh (scale sqrt2*sigma) + bg
        return A * (r / (2 * sigma ** 2)) * np.exp(-r ** 2 / (4 * sigma ** 2)) + B

    mode = centers[int(np.argmax(hist))]
    try:
        popt, _ = curve_fit(model, centers, hist, p0=[hist.max(), max(mode / 1.41, 1.0), 0.0],
                            bounds=([0, 1e-3, 0], [np.inf, max_d, np.inf]), maxfev=5000)
        return float(popt[1])
    except Exception:
        return float(mode / 1.41)                      # mode of the Rayleigh term = sqrt2*sigma


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    metric = p.pop("metric", "frc")
    p.pop("pixel_size_nm", None)
    p.pop("units", None)

    locs = read_localizations(input_csv)
    x = locs["x"].to_numpy(float)
    y = locs["y"].to_numpy(float)

    if metric == "nena":
        sigma = _nena(x, y, locs["frame"].to_numpy())
        print(f"nena: localization precision = {sigma:.1f} nm")
        return write_table(pd.DataFrame({"metric": ["nena_precision_nm"], "value": [sigma]}),
                           output_csv)

    # frc (default)
    render_nm = float(p.get("render_nm", 10.0))
    freq, frc, resolution = _frc(x, y, render_nm)
    op = Path(output_csv)
    write_table(pd.DataFrame({"spatial_freq_per_nm": freq, "frc": frc}),
                op.parent / "frc_curve.csv")
    print(f"frc: resolution = {resolution:.1f} nm")
    return write_table(pd.DataFrame({"metric": ["frc_resolution_nm"], "value": [resolution]}),
                       output_csv)
