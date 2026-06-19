#!/usr/bin/env python
"""
micro-SAM segmentation runner -- runs INSIDE the smlm-labflow/microsam image,
invoked by labflow's `runtime: docker` segment method over the file contract.

Contract: image in (TIFF) -> uint16 label mask out (TIFF). Uses micro_sam's
automatic instance segmentation (Segment Anything fine-tuned for microscopy; Archit
et al. 2024). The helper's signature has evolved across releases, so the call is
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
    model_type = p.get("model", "vit_b_lm")          # ViT-B fine-tuned for light microscopy
    try:
        from micro_sam.automatic_segmentation import (
            automatic_instance_segmentation, get_predictor_and_segmenter)
        predictor, segmenter = get_predictor_and_segmenter(model_type=model_type)
        masks = automatic_instance_segmentation(predictor, segmenter, input_path=img)
    except (ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(
            "micro-SAM API mismatch: " + repr(exc) +
            "\nAdjust docker/microsam/run_microsam.py to your installed micro_sam version "
            "(see micro_sam.automatic_segmentation).")

    tifffile.imwrite(args.out, np.asarray(masks).astype(np.uint16))
    print(f"micro-sam: segmented {int(np.asarray(masks).max())} objects -> {args.out}")


if __name__ == "__main__":
    main()
