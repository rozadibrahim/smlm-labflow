"""
labflow.stages.counting

Molecular counting from a clustered localization table. `method` (bind) selects:

    qpaint  - qPAINT (Jungmann et al., Nat. Methods 2016). In DNA-PAINT a docking
              site blinks each time an imager strand binds; the binding influx
              xi = 1 / <dark time> is proportional to the number of docking sites in
              a cluster. Dividing by the influx of a single site (a calibration,
              `unit_influx`) yields the molecule number. The influx itself (the
              "qPAINT index") is the calibration-free observable and is always
              reported; n_molecules is filled in only when unit_influx is given.
    ibfcs   - imaging FCS+ counting needs raw intensity time-traces, not a
              localization table, so it is not derivable from this contract.

Input : clusters.csv  (frame, x, y, cluster_id)        [cluster stage output]
Output: counts.csv     (cluster_id, n_localizations, mean_dark_frames,
                        qpaint_influx, n_molecules)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import read_table, write_table


def _qpaint(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    unit = float(params.get("unit_influx", 0.0))   # influx of ONE docking site (calib)
    rows = []
    for cid, g in df.groupby("cluster_id"):
        if int(cid) == -1:                          # noise label
            continue
        frames = np.unique(g["frame"].to_numpy(dtype=float))
        n_loc = int(len(g))
        if frames.size >= 2:
            dark = np.diff(frames)                   # dark intervals between bindings
            mean_dark = float(dark.mean())
            influx = 1.0 / mean_dark if mean_dark > 0 else float("nan")
        else:
            mean_dark, influx = float("nan"), float("nan")
        n_mol = influx / unit if (unit > 0 and influx == influx) else float("nan")
        rows.append({"cluster_id": int(cid), "n_localizations": n_loc,
                     "mean_dark_frames": mean_dark, "qpaint_influx": influx,
                     "n_molecules": n_mol})
    return pd.DataFrame(rows, columns=["cluster_id", "n_localizations",
                                       "mean_dark_frames", "qpaint_influx",
                                       "n_molecules"])


def _blink(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Blink-based molecular counting (PALM stoichiometry). Count blinking episodes per
    cluster -- consecutive frames are one blink; a gap > max_gap starts a new blink --
    then n_molecules = n_blinks / blinks_per_molecule (a photophysics calibration; the
    localization-derivable analogue of photobleaching-step counting)."""
    blinks_per_mol = float(params.get("blinks_per_molecule", 1.0))
    max_gap = int(params.get("max_gap", 1))
    rows = []
    for cid, g in df.groupby("cluster_id"):
        if int(cid) == -1:
            continue
        frames = np.sort(np.unique(g["frame"].to_numpy(dtype=int)))
        n_blinks = 1 + int(np.count_nonzero(np.diff(frames) > max_gap)) if frames.size else 0
        n_mol = n_blinks / blinks_per_mol if blinks_per_mol > 0 else float("nan")
        rows.append({"cluster_id": int(cid), "n_localizations": int(len(g)),
                     "n_blinks": int(n_blinks), "n_molecules": n_mol})
    return pd.DataFrame(rows, columns=["cluster_id", "n_localizations", "n_blinks", "n_molecules"])


def run(*, input_csv: str, output_csv: str, params: dict | None = None) -> str:
    params = dict(params or {})
    method = params.get("method", "qpaint")
    df = read_table(input_csv)
    if "cluster_id" not in df.columns:
        raise ValueError("counting needs a 'cluster_id' column — run the cluster stage first.")

    if method == "qpaint":
        out = _qpaint(df, params)
    elif method == "blink":
        out = _blink(df, params)
    elif method == "ibfcs":
        raise NotImplementedError(
            "ibfcs counting needs raw intensity time-traces (imaging FCS), not a "
            "localization/cluster table — it can't be derived from this contract.")
    else:
        raise ValueError(f"unknown counting method {method!r} (use 'qpaint' or 'blink').")

    return write_table(out, output_csv)
