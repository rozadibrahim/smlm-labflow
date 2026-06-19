"""
labflow.stages.cluster_voronoi

Voronoi-tessellation cluster calling (SR-Tesseler; Levet et al., Nat. Methods 2015).
Each localization's Voronoi cell area gives its local density (1/area); localizations
denser than `density_factor` x the average density are flagged 'in cluster', and
Voronoi-adjacent dense localizations are grouped into clusters (connected components
over shared Voronoi ridges). Pure numpy/scipy -> runs in the core env.

Contract: localizations CSV in -> clusters.csv out (localizations annotated with
`cluster_id`, noise = -1) + cluster_summary.csv (one row per cluster) alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..io import read_localizations, write_table


def _cell_areas(vor, n: int) -> np.ndarray:
    """Polygon area of each point's Voronoi cell (nan for unbounded cells)."""
    areas = np.full(n, np.nan)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:           # unbounded cell -> no finite density
            continue
        poly = vor.vertices[region]
        x, y = poly[:, 0], poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return areas


def _components(adjacency: Dict[int, list], dense_idx: np.ndarray, n: int) -> np.ndarray:
    """Connected components of dense localizations over the Voronoi adjacency graph."""
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cid = 0
    for seed in dense_idx:
        if visited[seed]:
            continue
        stack, comp = [int(seed)], []
        visited[seed] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adjacency.get(u, ()):
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        for u in comp:
            labels[u] = cid
        cid += 1
    return labels


def _drop_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Demote clusters below min_size to noise and relabel 0..K-1 contiguously."""
    out = np.full_like(labels, -1)
    keep, nxt = {}, 0
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    big = {int(u) for u, c in zip(uniq, counts) if c >= min_size}
    for i, c in enumerate(labels):
        c = int(c)
        if c in big:
            if c not in keep:
                keep[c] = nxt
                nxt += 1
            out[i] = keep[c]
    return out


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    p.pop("pixel_size_nm", None)
    p.pop("units", None)
    density_factor = float(p.get("density_factor", 2.0))
    min_cluster_size = int(p.get("min_cluster_size", 5))

    locs = read_localizations(input_csv)
    X = locs[["x", "y"]].to_numpy(float)
    n = len(X)
    labels = np.full(n, -1, dtype=int)

    if n >= 4:
        from scipy.spatial import Voronoi

        vor = Voronoi(X)
        areas = _cell_areas(vor, n)
        bbox = X.max(axis=0) - X.min(axis=0)
        avg_density = n / float(max(np.prod(bbox), 1e-9))
        density = np.where(np.isfinite(areas) & (areas > 0), 1.0 / areas, 0.0)
        dense = density > density_factor * avg_density

        adjacency: Dict[int, list] = {int(i): [] for i in np.flatnonzero(dense)}
        for a, b in vor.ridge_points:            # neighbouring localizations
            if dense[a] and dense[b]:
                adjacency[int(a)].append(int(b))
                adjacency[int(b)].append(int(a))
        labels = _components(adjacency, np.flatnonzero(dense), n)

    labels = _drop_small(labels, min_cluster_size)

    out = locs.copy()
    out["cluster_id"] = labels
    op = Path(output_csv)
    write_table(out, op)

    rows = []
    for cid in sorted({int(c) for c in labels if c >= 0}):
        mask = labels == cid
        pts = X[mask]
        centroid = pts.mean(axis=0)
        r_gyr = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1).mean()))
        rows.append({"cluster_id": cid, "n_localizations": int(mask.sum()),
                     "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
                     "radius_gyration_nm": r_gyr})
    write_table(pd.DataFrame(rows), op.parent / "cluster_summary.csv")

    n_clusters = len(rows)
    n_noise = int((labels < 0).sum())
    print(f"voronoi (SR-Tesseler): {n:,} localizations -> {n_clusters} clusters; "
          f"noise={n_noise:,}")
    return str(op)
