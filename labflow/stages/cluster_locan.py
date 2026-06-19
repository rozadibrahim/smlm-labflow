"""
labflow.stages.cluster_locan

Cluster calling via locan (the SMLM point-cloud library). DBSCAN/HDBSCAN through
locan returns rich per-cluster properties (localization count, centroid, bounding
-box area and density), so the cluster_summary is more informative than the
plain-sklearn backend.

Contract: localizations CSV in -> clusters.csv (localizations + cluster_id, noise
-1) + cluster_summary.csv (locan's per-cluster properties) alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..io import read_localizations, write_table


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    import locan as lc

    p = dict(params or {})
    algorithm = str(p.pop("algorithm", "dbscan")).lower()
    p.pop("pixel_size_nm", None)
    p.pop("units", None)

    locs = read_localizations(input_csv).reset_index(drop=True)
    df = pd.DataFrame({
        "position_x": locs["x"].to_numpy(float),
        "position_y": locs["y"].to_numpy(float),
        "frame": locs["frame"].to_numpy(),
    })
    ld = lc.LocData.from_dataframe(dataframe=df)

    if algorithm == "hdbscan":
        _noise, clusters = lc.cluster_hdbscan(
            ld, min_cluster_size=int(p.get("min_cluster_size", 20)))
    else:
        _noise, clusters = lc.cluster_dbscan(
            ld, eps=float(p.get("eps", 100.0)), min_samples=int(p.get("min_samples", 10)))

    refs = clusters.references or []
    labels = np.full(len(locs), -1, dtype=int)
    for cid, ref in enumerate(refs):
        idx = np.asarray(ref.data.index, dtype=int)
        labels[idx[(idx >= 0) & (idx < len(locs))]] = cid

    out = locs.copy()
    out["cluster_id"] = labels
    op = Path(output_csv)
    write_table(out, op)

    summary = clusters.data.copy()
    summary.insert(0, "cluster_id", range(len(summary)))
    write_table(summary, op.parent / "cluster_summary.csv")

    print(f"locan {algorithm}: {len(refs)} clusters, "
          f"noise={int((labels < 0).sum()):,} / {len(locs):,} localizations")
    return str(op)
