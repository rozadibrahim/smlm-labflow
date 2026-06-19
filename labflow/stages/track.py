"""
labflow.stages.track

In-process trajectory linking with trackpy (Crocker-Grier). Links localizations
across frames into trajectories.

Contract: localizations CSV in -> tracks.csv out (track_id, frame, x, y[, z]).
(The other track backends -- MAGIK, TrackMate, Spot-On -- are subprocess/GUI
methods declared separately in the registry.)
"""

from __future__ import annotations

from typing import Any, Dict

from ..io import read_localizations, write_table


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    import trackpy as tp

    p = dict(params or {})
    p.pop("method", None)
    p.pop("pixel_size_nm", None)
    p.pop("units", None)
    search_range = float(p.get("search_range", 500.0))
    memory = int(p.get("memory", 3))

    locs = read_localizations(input_csv)          # frame, x, y[, z]
    cols = ["frame", "x", "y"] + (["z"] if "z" in locs.columns else [])
    f = locs[cols].reset_index(drop=True)

    tp.quiet()
    linked = tp.link(f, search_range=search_range, memory=memory)
    linked = linked.rename(columns={"particle": "track_id"}).sort_values(["track_id", "frame"])

    keep = ["track_id"] + cols
    n_tracks = int(linked["track_id"].nunique())
    print(f"trackpy linked {len(linked):,} localizations -> {n_tracks} tracks "
          f"(search_range={search_range} nm, memory={memory})")
    return write_table(linked[keep], output_csv)
