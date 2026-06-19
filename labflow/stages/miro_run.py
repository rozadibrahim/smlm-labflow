#!/usr/bin/env python
"""
labflow.stages.miro_run

Runner for the MIRO deep-learning cluster backend. Runs INSIDE the `miro` env
(torch + deeplay + MIRO), invoked by the registry as a `runtime: venv` method.

MIRO (DeepTrackAI, "Enhanced spatial clustering of single-molecule localizations
with graph neural networks", Nat. Commun. 2025; github.com/DeepTrackAI/MIRO) is a
recurrent graph-NN that *transforms* the localization point cloud so that a
conventional density clusterer (DBSCAN) separates nanoclusters far better. So the
pipeline here is: load localizations -> MIRO transform -> DBSCAN -> clusters.csv,
matching the labflow cluster contract (same outputs as the sklearn backends).

The IO + DBSCAN + summary are implemented; the single MIRO-specific call is an
explicit binding point (`miro_transform`) because MIRO's exact API lives in its
tutorial notebooks and depends on a trained model — bind it in your `miro` env.
It raises a clear error rather than fabricating results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def miro_transform(coords: np.ndarray, params: dict) -> np.ndarray:
    """Transform the point cloud with a trained MIRO model -> compact coordinates.

    BINDING POINT. In the `miro` env this should:
      1. import the MIRO/deeplay API,
      2. load the trained model at `params["model"]`,
      3. build the localization graph from `coords` and return transformed coords
         (same N, the rGNN-compacted positions DBSCAN then clusters).
    See the MIRO tutorial notebooks (Benchmark / Multiscale / Multishape).
    """
    try:
        import miro  # noqa: F401  (provided by the miro env)
    except Exception as exc:
        raise SystemExit(
            "MIRO is not importable in this environment. Create the 'miro' env:\n"
            "    git clone https://github.com/DeepTrackAI/MIRO\n"
            "    python -m venv envs/miro && envs/miro/Scripts/pip install -r MIRO/requirements.txt\n"
            "(installs deeplay/torch), then bind miro_transform() to the MIRO model API."
        ) from exc

    model_path = params.get("model") or ""
    if not model_path:
        raise SystemExit(
            "MIRO needs a trained model: pass --param model=/path/to/miro_model "
            "(see the MIRO tutorials to train/obtain one), then bind miro_transform()."
        )
    raise SystemExit(
        "miro_transform() is not bound yet — implement the MIRO model load + transform "
        "in your miro env (github.com/DeepTrackAI/MIRO tutorials). The rest of this "
        "runner (DBSCAN + outputs) is ready."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    params = json.loads(args.params)

    df = pd.read_csv(args.inp)
    cols = {c.lower(): c for c in df.columns}
    xc = cols.get("x") or cols.get("x [nm]")
    yc = cols.get("y") or cols.get("y [nm]")
    X = df[[xc, yc]].to_numpy(float)

    Xt = miro_transform(X, params)        # MIRO rGNN transform (binding point)

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=float(params.get("eps", 1.0)),
                    min_samples=int(params.get("min_samples", 10))).fit_predict(Xt)

    out = df.copy()
    out["cluster_id"] = labels
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(op, index=False)

    rows = []
    for cid in sorted(set(int(c) for c in labels)):
        if cid < 0:
            continue
        m = labels == cid
        c = X[m].mean(axis=0)
        rows.append({"cluster_id": cid, "n_localizations": int(m.sum()),
                     "centroid_x": float(c[0]), "centroid_y": float(c[1]),
                     "radius_gyration_nm": float(np.sqrt(((X[m] - c) ** 2).sum(1).mean()))})
    pd.DataFrame(rows).to_csv(op.parent / "cluster_summary.csv", index=False)


if __name__ == "__main__":
    main()
