#!/usr/bin/env python
"""
Cellpose segmentation runner — runs INSIDE the smlm-labflow/cellpose image,
invoked by labflow's `runtime: docker` method over the file contract.

Contract: image in (TIFF) -> mask label image out (uint16 TIFF), the labflow
`segment` stage output. Pinned to the Cellpose 3.x API; if you change the
Cellpose version, verify the `models.CellposeModel(...).eval(...)` call.
"""

import argparse
import json

import numpy as np
import tifffile
from cellpose import models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    p = json.loads(args.params)

    img = tifffile.imread(args.inp)
    model = models.CellposeModel(gpu=False, model_type=p.get("model_type", "cyto3"))
    diameter = p.get("diameter", 0) or None     # 0/None -> auto-estimate
    masks, _flows, _styles = model.eval(img, diameter=diameter)

    tifffile.imwrite(args.out, masks.astype(np.uint16))
    print(f"cellpose: segmented {int(masks.max())} objects -> {args.out}")


if __name__ == "__main__":
    main()
