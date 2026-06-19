"""
Bayesian cluster-analysis adapter (Griffie / Rubin-Delanchy / Owen 2016: a Bayesian,
model-based cluster identification for SMLM) -- runs in the `bayesian_cluster` venv.

Contract: localizations.csv (frame, x, y) in -> clusters.csv (localizations + cluster_id) out.

BINDING POINT: the reference method scores clustering proposals with a Bayesian model
(Ripley-K based) and is distributed as MATLAB/R. Wire `_cluster()` to your port (or
implement the scoring). The IO + cluster contract are done here.
"""

import argparse
import json

import numpy as np  # noqa: F401  (for the binding-point implementation)
import pandas as pd


def _cluster(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return df with a `cluster_id` column (noise = -1). Bayesian-model binding point."""
    raise NotImplementedError(
        "Bayesian cluster scoring is the binding point: wire adapters/bayesian_cluster.py:"
        "_cluster to your port of the Griffie/Owen model. IO is done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    df = _cluster(pd.read_csv(args.inp), json.loads(args.params))
    cols = [c for c in ("frame", "x", "y", "z") if c in df.columns] + ["cluster_id"]
    df[cols].to_csv(args.out, index=False)
    print(f"bayesian: {df['cluster_id'].nunique()} clusters -> {args.out}")


if __name__ == "__main__":
    main()
