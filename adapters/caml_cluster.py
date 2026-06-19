"""
CAML cluster-analysis adapter (Williamson et al., Nat. Methods 2020: "Machine learning
for cluster analysis of localization microscopy data") -- runs in the `caml` venv.

Contract: localizations.csv (frame, x, y) in -> clusters.csv (localizations + cluster_id) out.

BINDING POINT: CAML uses trained neural-network models to classify/segment clusters; its
scripts + models are user-provided (set install.git, or drop them in envs/caml_src). Wire
`_cluster()` to your CAML checkout + model. The IO + cluster contract are done here.
"""

import argparse
import json
import os
import sys

import numpy as np  # noqa: F401  (for the binding-point implementation)
import pandas as pd


def _cluster(df: pd.DataFrame, params: dict, src: str) -> pd.DataFrame:
    """Return df with a `cluster_id` column (noise = -1). CAML model binding point."""
    if src and os.path.isdir(src):
        sys.path.insert(0, src)
    raise NotImplementedError(
        "CAML model is the binding point: wire adapters/caml_cluster.py:_cluster to your "
        "CAML checkout + trained model (returns a cluster_id per localization). IO is done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--src", default="")
    args = ap.parse_args()

    df = _cluster(pd.read_csv(args.inp), json.loads(args.params), args.src)
    cols = [c for c in ("frame", "x", "y", "z") if c in df.columns] + ["cluster_id"]
    df[cols].to_csv(args.out, index=False)
    print(f"caml: {df['cluster_id'].nunique()} clusters -> {args.out}")


if __name__ == "__main__":
    main()
