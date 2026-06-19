#!/usr/bin/env python
"""
Omnipose segmentation runner -- runs INSIDE the smlm-labflow/omnipose image,
invoked by labflow's `runtime: docker` segment method over the file contract.

Contract: image in (TIFF) -> uint16 label mask out (TIFF). Omnipose (Cutler et al.
2022) targets elongated / bacterial cells and ships a patched cellpose
(`cellpose_omni`); its eval signature has shifted across versions, so the call is
guarded -- on a mismatch it fails with an actionable message rather than wrong masks.
"""

import argparse
import json

import numpy as np
import tifffile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    p = json.loads(args.params)

    img = tifffile.imread(args.inp)
    model_type = p.get("model_type", "bact_phase_omni")
    try:
        try:
            from cellpose_omni import models           # omnipose's patched fork
        except ImportError:
            from cellpose import models                 # older / combined layout
        model = models.CellposeModel(gpu=False, model_type=model_type)
        result = model.eval(img, channels=p.get("channels", [0, 0]), omni=True)
        masks = result[0]
    except (ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(
            "omnipose API mismatch: " + repr(exc) +
            "\nAdjust docker/omnipose/run_omnipose.py to your installed omnipose version.")

    tifffile.imwrite(args.out, np.asarray(masks).astype(np.uint16))
    print(f"omnipose: segmented {int(np.asarray(masks).max())} objects -> {args.out}")


if __name__ == "__main__":
    main()
