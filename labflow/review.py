"""
labflow.review

Open a run in napari for interactive inspection — launched *through* labflow, but
showing the full napari GUI (the one deliberate GUI exception). Read-only review:
localizations, drift-corrected localizations, and clusters become napari layers.

`gather_layers` is GUI-free (and unit-tested); `open_in_napari` launches the real
napari viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _find(run_dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    for n in names:                       # else search batches one+ levels down
        hits = sorted(run_dir.glob(f"**/{n}"))
        if hits:
            return hits[0]
    return None


def _xy(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    def pick(opts):
        for o in opts:
            if o in cols:
                return cols[o]
        return None

    xc = pick(["x", "x [nm]", "x_nm"])
    yc = pick(["y", "y [nm]", "y_nm"])
    if xc is None or yc is None:
        raise ValueError(f"no x/y columns in {list(df.columns)}")
    return df[xc].to_numpy(float), df[yc].to_numpy(float)


def gather_layers(run_dir: str | Path) -> List[Dict[str, Any]]:
    """Build napari layer specs from a run dir, or from a single CSV (GUI-free)."""
    run_dir = Path(run_dir)

    if run_dir.is_file():                       # a single localizations/clusters CSV
        df = pd.read_csv(run_dir)
        x, y = _xy(df)
        layer: Dict[str, Any] = {"name": run_dir.name, "type": "points",
                                 "data": np.column_stack([y, x])}
        if "cluster_id" in df.columns:
            layer["cluster_id"] = df["cluster_id"].to_numpy()
        return [layer]

    layers: List[Dict[str, Any]] = []
    loc = _find(run_dir, ["drift_corrected_localizations.csv",
                          "canonical_localizations.csv", "localizations.csv"])
    if loc is not None:
        x, y = _xy(pd.read_csv(loc))
        layers.append({"name": f"localizations ({loc.name})", "type": "points",
                       "data": np.column_stack([y, x])})   # napari = (row=y, col=x)

    clu = _find(run_dir, ["clusters.csv"])
    if clu is not None:
        df = pd.read_csv(clu)
        x, y = _xy(df)
        cid = df["cluster_id"].to_numpy() if "cluster_id" in df.columns else None
        layers.append({"name": "clusters", "type": "points",
                       "data": np.column_stack([y, x]), "cluster_id": cid})

    return layers


def open_in_napari(run_dir: str | Path):
    try:
        import napari
    except Exception as exc:
        raise RuntimeError(
            "napari is not installed. Install it to use `labflow review`:\n"
            '    pip install "napari[all]"'
        ) from exc

    layers = gather_layers(run_dir)
    if not layers:
        raise RuntimeError(
            f"No reviewable data found under {run_dir} "
            "(expected localizations / clusters CSVs)."
        )

    viewer = napari.Viewer(title=f"labflow review · {Path(run_dir).name}")
    for layer in layers:
        kwargs: Dict[str, Any] = {"name": layer["name"], "size": 6}
        if layer.get("cluster_id") is not None:
            kwargs["features"] = {"cluster_id": layer["cluster_id"]}
            kwargs["face_color"] = "cluster_id"
        viewer.add_points(layer["data"], **kwargs)
    napari.run()                          # full interactive napari GUI
    return viewer
