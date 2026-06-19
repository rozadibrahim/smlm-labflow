"""
FD-DeepLoc localizer adapter  (Fu et al., Nat. Commun. 2023; Li-Lab-SUSTech/FD-DeepLoc).

Runs INSIDE the `fd_deeploc` conda env (built by `labflow install fd_deeploc`, which
also clones the repo to envs/fd_deeploc_src). Reads a frame stack, runs FD-DeepLoc
inference (with its field-dependent aberration model), and writes canonical
localizations; labflow validates the contract on the core side.

    labflow run localize -b fd_deeploc -i frames.tif -o locs.csv \
        --param model=/path/trained_model --param calibration=/path/psf_calib

BINDING POINT
-------------
FD-DeepLoc is research code (no stable PyPI API), so the repo is cloned and added to
sys.path. The IO here is complete and canonical; the inference call is the one piece
to wire to your checkout (its demo notebooks show the exact evaluator). It raises a
clear NotImplementedError until wired -- no fabricated API, no wrong coordinates.
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
        raise SystemExit("FD-DeepLoc needs --param model=<trained_model> "
                         "(and a calibrated field-dependent PSF via --param calibration=).")
    if not src or not os.path.isdir(src):
        raise SystemExit(f"FD-DeepLoc source not found at {src!r}; "
                         "run `labflow install fd_deeploc` first.")
    sys.path.insert(0, src)

    # --- BINDING POINT: connect to the cloned FD-DeepLoc evaluator --------------
    # Open the repo's demo notebooks (envs/fd_deeploc_src) and wire its inference,
    # roughly:
    #     from fd_deeploc_core import <Evaluator/infer>   # confirm names in your checkout
    #     model = load(model_path)
    #     preds = evaluate(model, frames, params)          # field-dependent
    # then return arrays: frame_ix, x_nm, y_nm, z_nm, photons.
    raise NotImplementedError(
        "FD-DeepLoc inference is the binding point. Wire adapters/fd_deeploc_localize.py:"
        "_infer to envs/fd_deeploc_src (the cloned repo's eval routine + your trained model). "
        "The frame IO and canonical CSV output around it are already done.")


def main():
    ap = argparse.ArgumentParser(description="FD-DeepLoc localizer adapter (labflow).")
    ap.add_argument("--in", dest="inp", required=True, help="frame stack (TIFF)")
    ap.add_argument("--out", dest="out", required=True, help="canonical localizations CSV")
    ap.add_argument("--params", default="{}", help="JSON params (model, calibration)")
    ap.add_argument("--src", default="", help="cloned FD-DeepLoc source dir")
    args = ap.parse_args()

    params = json.loads(args.params) if args.params else {}
    frames = read_frames(args.inp)
    frame, x, y, z, photons = _infer(frames, params, args.src)
    write_localizations(args.out, frame=frame, x=x, y=y, z=z, photons=photons,
                        backend="fd_deeploc", source_file=args.inp)
    print(f"fd_deeploc: {len(frame)} localizations -> {args.out}")


if __name__ == "__main__":
    main()
