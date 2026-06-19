"""
Spot-On diffusion analysis adapter (Hansen et al. 2018) -- runs in the `spot_on` venv.

NOTE: Spot-On is NOT a linker. It analyzes already-linked trajectories to fit a 2-3
state kinetic model (bound/free fractions + diffusion coefficients). So it consumes
tracks, the tracks PASS THROUGH unchanged (keeping the `track` file contract valid),
and the Spot-On result is written alongside as spoton_kinetics.csv.

Contract: tracks.csv (track_id, frame, x, y) in -> tracks.csv out (unchanged) +
spoton_kinetics.csv alongside.

BINDING POINT: the 2-state fit uses fastspt (the engine behind Spot-On); wire `_fit()`
to your fastspt install. The pass-through keeps the pipeline valid until then.
"""

import argparse
import json
import os

import pandas as pd


def _fit(tracks: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return a small kinetics table (f_bound, D_bound, D_free, ...). fastspt binding point."""
    # import fastspt
    # jld = fastspt.compute_jump_length_distribution(tracks, ...)
    # fit = fastspt.fit_jump_length_distribution(jld, **params)
    # return pd.DataFrame([fit])
    raise NotImplementedError(
        "Spot-On 2-state fit is the binding point: wire adapters/spot_on_track.py:_fit "
        "to fastspt. Tracks pass through; the kinetics file needs this.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    tracks = pd.read_csv(args.inp)
    tracks.to_csv(args.out, index=False)          # pass-through: track contract stays valid
    try:
        kinetics = _fit(tracks, json.loads(args.params))
        dest = os.path.join(os.path.dirname(args.out) or ".", "spoton_kinetics.csv")
        kinetics.to_csv(dest, index=False)
        print(f"spot_on: kinetics -> {dest}; tracks passed through -> {args.out}")
    except NotImplementedError as exc:
        print(f"spot_on: tracks passed through -> {args.out} | "
              f"kinetics pending: {str(exc).splitlines()[0]}")


if __name__ == "__main__":
    main()
