#!/usr/bin/env python
"""
MAGIK GNN tracking runner -- runs INSIDE the smlm-labflow/magik image (DeepTrackAI,
Pineda et al. 2023), invoked by labflow's `runtime: docker` track method.

Contract: localizations.csv (frame, x, y[, z]) in -> tracks.csv (track_id, frame,
x, y[, z]) out, the labflow `track` stage output.

BINDING POINT: MAGIK is a graph neural network whose inference is notebook-driven and
needs a trained model; its exact API is not stable across releases. The IO + the
tracks.csv contract here are complete -- wire `_link()` to the DeepTrack/MAGIK GNN in
your image. It raises a clear message until then (no fabricated API, no wrong tracks).
"""

import argparse
import json

import numpy as np  # noqa: F401  (available for the binding-point implementation)
import pandas as pd


def _link(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return df with a `track_id` column. MAGIK GNN binding point."""
    # Build the spatiotemporal graph and run the trained MAGIK model, e.g.:
    #     from deeptrack.models.gnns import ...     # confirm names in your install
    #     graph = build_graph(df, connectivity_radius=params.get("connectivity_radius"))
    #     pred  = model(graph); df["track_id"] = assign_tracks(pred)
    raise NotImplementedError(
        "MAGIK inference is the binding point: wire docker/magik/run_magik.py:_link to "
        "the DeepTrackAI MAGIK GNN + a trained model. IO + tracks.csv output are done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    linked = _link(df, json.loads(args.params))
    cols = ["track_id", "frame", "x", "y"] + (["z"] if "z" in linked.columns else [])
    linked[cols].to_csv(args.out, index=False)
    print(f"magik: {linked['track_id'].nunique()} tracks -> {args.out}")


if __name__ == "__main__":
    main()
