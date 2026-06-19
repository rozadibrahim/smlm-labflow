"""
labflow.stages.spatial_stats

In-process spatial point-pattern statistics on localizations. One adapter; the
`metric` param selects Ripley's K/L, the pair-correlation g(r), nearest-neighbour
distances, or Voronoi densities. numpy + scikit-learn neighbours (scipy for
Voronoi).

Contract: localizations CSV in -> spatial_stats.csv out (a curve r vs statistic,
or a per-localization distribution depending on the metric).

Note: the radius-based metrics are O(N x neighbours); fine to ~10^4-10^5 points.
At full SMLM scale this is the kind of pairwise kernel that benefits from a Julia
backend (see docs) -- a future drop-in.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..io import read_localizations, write_table


def _bbox_area(X: np.ndarray) -> float:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return float(max(np.prod(maxs - mins), 1e-9))


def _radius_curves(X, r_max, n_bins):
    """Ripley K/L and pair-correlation g(r) from pairwise distances < r_max."""
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    area = _bbox_area(X)
    density = n / area
    nn = NearestNeighbors().fit(X)
    dist_lists = nn.radius_neighbors(X, radius=r_max, return_distance=True)[0]
    d = np.concatenate([a[a > 0] for a in dist_lists]) if n else np.array([])

    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    hist, _ = np.histogram(d, bins=edges)        # ordered pairs per annulus

    cum = np.cumsum(hist)
    K = area / (n * n) * cum
    L = np.sqrt(np.clip(K, 0, None) / np.pi)
    expected = n * density * 2 * np.pi * centers * dr
    g = np.divide(hist, expected, out=np.zeros_like(centers), where=expected > 0)
    return centers, K, L, g


def _voronoi_areas(X):
    from scipy.spatial import Voronoi

    vor = Voronoi(X)
    areas = np.full(len(X), np.nan)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:           # unbounded cell -> skip
            continue
        poly = vor.vertices[region]
        x, y = poly[:, 0], poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return areas


def _channel2(input_csv, p):
    """Second-colour coordinates: channel2=<path> for 2-colour, else channel1 (self)."""
    return read_localizations(p.get("channel2") or input_csv)[["x", "y"]].to_numpy(float)


def _cbc(A, B, r_max, n_bins):
    """Coordinate-based colocalization (Malkusch et al. 2012): per-A localization C in
    [-1, 1] (+1 colocalized, -1 segregated). Spearman correlation of the density-corrected
    A and B neighbour distributions around each A localization, weighted by the distance
    to the nearest B localization."""
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    treeA, treeB = cKDTree(A), cKDTree(B)
    radii = np.linspace(0.0, r_max, n_bins + 1)[1:]
    norm = r_max ** 2 / radii ** 2
    nn_B, _ = treeB.query(A, k=1)
    out = np.zeros(len(A))
    for i, pt in enumerate(A):
        dA = np.sort(np.linalg.norm(A[treeA.query_ball_point(pt, r_max)] - pt, axis=1))
        dB = np.sort(np.linalg.norm(B[treeB.query_ball_point(pt, r_max)] - pt, axis=1))
        NA = np.searchsorted(dA, radii, side="right").astype(float)
        NB = np.searchsorted(dB, radii, side="right").astype(float)
        DA = (NA / max(NA[-1], 1.0)) * norm
        DB = (NB / max(NB[-1], 1.0)) * norm
        rho = spearmanr(DA, DB).correlation if (DA.std() > 0 and DB.std() > 0) else 0.0
        out[i] = (rho if np.isfinite(rho) else 0.0) * np.exp(-nn_B[i] / r_max)
    return out


def _cross_corr(A, B, r_max, n_bins):
    """Two-colour cross-correlation g_AB(r): B density around A, normalized to CSR."""
    from scipy.spatial import cKDTree

    treeB = cKDTree(B)
    mins = np.minimum(A.min(0), B.min(0))
    maxs = np.maximum(A.max(0), B.max(0))
    area = float(max(np.prod(maxs - mins), 1e-9))
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    hist = np.zeros(n_bins)
    for pt in A:
        idx = treeB.query_ball_point(pt, r_max)
        if idx:
            hist += np.histogram(np.linalg.norm(B[idx] - pt, axis=1), bins=edges)[0]
    expected = len(A) * (len(B) / area) * 2 * np.pi * centers * dr
    g = np.divide(hist, expected, out=np.zeros_like(centers), where=expected > 0)
    return centers, g


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    metric = p.pop("metric", "ripley")
    p.pop("pixel_size_nm", None)
    p.pop("units", None)

    locs = read_localizations(input_csv)
    X = locs[["x", "y"]].to_numpy(float)

    if metric == "nnd":
        from sklearn.neighbors import NearestNeighbors
        d, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
        nnd = d[:, 1]
        print(f"nnd: {len(X):,} localizations, median NND {np.median(nnd):.1f} nm")
        return write_table(pd.DataFrame({"nnd_nm": nnd}), output_csv)

    if metric == "voronoi":
        try:
            areas = _voronoi_areas(X)
        except ImportError as exc:
            raise RuntimeError("voronoi metric needs scipy (`pip install scipy`).") from exc
        dens = np.where(areas > 0, 1.0 / areas, np.nan)
        finite = np.isfinite(areas).sum()
        print(f"voronoi: {finite:,}/{len(X):,} bounded cells")
        return write_table(
            pd.DataFrame({"voronoi_area_nm2": areas, "local_density_per_nm2": dens}),
            output_csv)

    if metric == "gfunction":
        from sklearn.neighbors import NearestNeighbors
        nnd = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
        centers = np.linspace(0.0, float(p.get("r_max", 500.0)), int(p.get("n_bins", 50)) + 1)[1:]
        gvals = np.array([(nnd <= r).mean() for r in centers])
        print(f"gfunction: {len(X):,} localizations")
        return write_table(pd.DataFrame({"r_nm": centers, "G": gvals}), output_csv)

    if metric in ("cbc", "crosscorrelation"):
        B = _channel2(input_csv, p)
        r_max = float(p.get("r_max", 500.0))
        if metric == "cbc":
            out = locs.copy()
            out["cbc"] = _cbc(X, B, r_max, int(p.get("n_bins", 20)))
            kind = "2-colour" if p.get("channel2") else "self"
            print(f"cbc ({kind}): {len(X):,} localizations, mean C={out['cbc'].mean():.3f}")
            return write_table(out, output_csv)
        r, gx = _cross_corr(X, B, r_max, int(p.get("n_bins", 50)))
        print(f"crosscorrelation: {len(X):,} x {len(B):,}, r_max {r_max:.0f} nm")
        return write_table(pd.DataFrame({"r_nm": r, "g_cross": gx}), output_csv)

    r_max = float(p.get("r_max", 500.0))
    n_bins = int(p.get("n_bins", 50))
    r, K, L, g = _radius_curves(X, r_max, n_bins)
    if metric == "paircorrelation":
        print(f"pair-correlation: {len(X):,} localizations, r_max {r_max:.0f} nm")
        return write_table(pd.DataFrame({"r_nm": r, "g": g}), output_csv)
    # ripley (default)
    print(f"ripley: {len(X):,} localizations, r_max {r_max:.0f} nm")
    return write_table(pd.DataFrame({"r_nm": r, "K": K, "L": L, "L_minus_r": L - r}),
                       output_csv)
