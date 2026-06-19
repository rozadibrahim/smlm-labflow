#!/usr/bin/env python
"""
StarDist segmentation runner -- runs INSIDE the smlm-labflow/stardist image,
invoked by labflow's `runtime: docker` segment method over the file contract.

Contract: image in (TIFF) -> uint16 label mask out (TIFF), the labflow `segment`
stage output. Pinned to the stable StarDist2D.predict_instances API (Schmidt 2018).
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

    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    img = tifffile.imread(args.inp)
    model = StarDist2D.from_pretrained(p.get("model", "2D_versatile_fluo"))
    labels, _ = model.predict_instances(normalize(img))

    tifffile.imwrite(args.out, labels.astype(np.uint16))
    print(f"stardist: segmented {int(labels.max())} objects -> {args.out}")


if __name__ == "__main__":
    main()
