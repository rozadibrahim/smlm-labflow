"""
DeepSTORM3D localizer adapter  (Nehme et al., Nat. Methods 2020; EliasNehme/DeepSTORM3D).

Runs INSIDE the `deepstorm3d` conda env (built by `labflow install deepstorm3d`, which
also clones the repo to envs/deepstorm3d_src). Reads a dense frame stack, runs the
trained CNN to recover 3D positions, and writes canonical localizations; labflow
validates the contract on the core side.

    labflow run localize -b deepstorm3d -i frames.tif -o locs.csv \
        --param model=/path/trained_model

BINDING POINT
-------------
DeepSTORM3D is research code (no stable PyPI API), so the repo is cloned and added to
sys.path. The IO here is complete and canonical; the inference call is the one piece
to wire to your checkout (its testing script shows the exact routine, e.g.
DeepSTORM3D.Testing). It raises a clear NotImplementedError until wired -- no
fabricated API, no wrong coordinates.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _localize_io import read_frames, write_localizations


def _infer(frames, params, src):
    """Return per-localization arrays (frame_ix, x_nm, y_nm, z_nm, photons)."""
    model_path = params.get("model") or ""
    if not model_path:
        raise SystemExit("DeepSTORM3D needs --param model=<trained_model>.")
    if not src or not os.path.isdir(src):
        raise SystemExit(f"DeepSTORM3D source not found at {src!r}; "
                         "run `labflow install deepstorm3d` first.")
    sys.path.insert(0, src)

    # --- BINDING POINT: connect to the cloned DeepSTORM3D testing routine -------
    # In the repo (envs/deepstorm3d_src), inference is in its testing script, roughly:
    #     from DeepSTORM3D.Testing import test_model      # confirm in your checkout
    #     xyz, photons = test_model(setup_params, frames, model_path)
    # then return arrays: frame_ix, x_nm, y_nm, z_nm, photons.
    raise NotImplementedError(
        "DeepSTORM3D inference is the binding point. Wire adapters/deepstorm3d_localize.py:"
        "_infer to envs/deepstorm3d_src (the cloned repo's testing routine + trained model). "
        "The frame IO and canonical CSV output around it are already done.")


def main():
    ap = argparse.ArgumentParser(description="DeepSTORM3D localizer adapter (labflow).")
    ap.add_argument("--in", dest="inp", required=True, help="frame stack (TIFF)")
    ap.add_argument("--out", dest="out", required=True, help="canonical localizations CSV")
    ap.add_argument("--params", default="{}", help="JSON params (model)")
    ap.add_argument("--src", default="", help="cloned DeepSTORM3D source dir")
    args = ap.parse_args()

    params = json.loads(args.params) if args.params else {}
    frames = read_frames(args.inp)
    frame, x, y, z, photons = _infer(frames, params, args.src)
    write_localizations(args.out, frame=frame, x=x, y=y, z=z, photons=photons,
                        backend="deepstorm3d", source_file=args.inp)
    print(f"deepstorm3d: {len(frame)} localizations -> {args.out}")


if __name__ == "__main__":
    main()
