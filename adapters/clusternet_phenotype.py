"""
ClusterNet phenotyping adapter (graph neural network for nanocluster classification)
-- runs in the `clusternet` venv. Builds a graph per cluster from its localizations
and classifies the cluster's phenotype with a trained GNN.

Contract: clusters.csv (localizations + cluster_id) in -> phenotypes.csv (cluster_id +
phenotype) out, the labflow `phenotype` stage output.

BINDING POINT: ClusterNet is a deep-learning graph classifier whose API + trained
weights are user-provided. The IO + the phenotype contract are done here -- wire
`_classify()` to your ClusterNet checkout + model (set install.git, or drop it in
envs/clusternet_src). It raises a clear message until then (no fabricated API).
"""

import argparse
import json
import os
import sys

import numpy as np  # noqa: F401  (for the binding-point implementation)
import pandas as pd


def _classify(df: pd.DataFrame, params: dict, src: str) -> pd.DataFrame:
    """Return one row per cluster: (cluster_id, phenotype). ClusterNet GNN binding point."""
    if src and os.path.isdir(src):
        sys.path.insert(0, src)
    raise NotImplementedError(
        "ClusterNet GNN is the binding point: wire adapters/clusternet_phenotype.py:_classify "
        "to your ClusterNet checkout + trained model (per-cluster graph -> phenotype). IO is done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--src", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "cluster_id" not in df.columns:
        raise SystemExit("phenotype needs a 'cluster_id' column - run the cluster stage first.")
    out = _classify(df, json.loads(args.params), args.src)
    out.to_csv(args.out, index=False)
    print(f"clusternet: phenotyped {out['cluster_id'].nunique()} clusters -> {args.out}")


if __name__ == "__main__":
    main()
