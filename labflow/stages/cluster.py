"""
labflow.stages.cluster

In-process cluster calling on localizations (structural SMLM analysis). One
adapter; the `algorithm` param selects DBSCAN / OPTICS / HDBSCAN (scikit-learn),
so all three dock through a single registry entry pattern.

Contract: localizations CSV in -> clusters.csv out (the localizations annotated
with `cluster_id`, noise = -1), with cluster_summary.csv (one row per cluster,
with size and radius of gyration) written alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..io import read_localizations, write_table


def _model(algorithm: str, p: Dict[str, Any]):
    algo = str(algorithm).lower()
    if algo == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(p.get("eps", 100.0)),
                      min_samples=int(p.get("min_samples", 10)))
    if algo == "optics":
        from sklearn.cluster import OPTICS
        return OPTICS(min_samples=int(p.get("min_samples", 10)),
                      max_eps=float(p.get("max_eps", np.inf)))
    if algo == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:
            raise RuntimeError(
                "HDBSCAN needs scikit-learn>=1.3 (`pip install -U scikit-learn`)."
            ) from exc
        return HDBSCAN(
            min_cluster_size=int(p.get("min_cluster_size", 20)),
            min_samples=(int(p["min_samples"]) if p.get("min_samples") is not None else None),
        )
    raise ValueError(f"unknown cluster algorithm {algorithm!r} (dbscan|optics|hdbscan)")


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    algorithm = p.pop("algorithm", "dbscan")
    p.pop("pixel_size_nm", None)          # not used by clustering itself
    p.pop("units", None)

    locs = read_localizations(input_csv)
    use_z = "z" in locs.columns and np.isfinite(locs["z"].to_numpy(float)).any()
    coords = ["x", "y", "z"] if use_z else ["x", "y"]
    X = locs[coords].to_numpy(float)

    labels = _model(algorithm, p).fit_predict(X)

    out = locs.copy()
    out["cluster_id"] = labels
    op = Path(output_csv)
    write_table(out, op)

    # per-cluster summary (noise label -1 excluded)
    rows = []
    for cid in sorted(set(int(c) for c in labels)):
        if cid < 0:
            continue
        mask = labels == cid
        pts = X[mask]
        centroid = pts.mean(axis=0)
        r_gyr = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1).mean()))
        row = {"cluster_id": cid, "n_localizations": int(mask.sum()),
               "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
               "radius_gyration_nm": r_gyr}
        if use_z:
            row["centroid_z"] = float(centroid[2])
        rows.append(row)
    write_table(pd.DataFrame(rows), op.parent / "cluster_summary.csv")

    n_clusters = sum(1 for c in set(int(c) for c in labels) if c >= 0)
    n_noise = int((labels < 0).sum())
    print(f"clustered {len(locs):,} localizations -> {n_clusters} clusters "
          f"({algorithm}); noise={n_noise:,}")
    return str(op)
