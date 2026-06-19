"""
Regression test for the Voronoi (SR-Tesseler) cluster backend.

Run: python -m pytest labflow/tests/test_cluster_voronoi.py
(or:  python labflow/tests/test_cluster_voronoi.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.stages.cluster_voronoi import run


def _make(tmp, n_blobs=3, per_blob=300, noise=400):
    rng = np.random.default_rng(0)
    centers = [(5000, 5000), (15000, 5000), (10000, 15000)][:n_blobs]
    pts = [rng.normal((cx, cy), 60, size=(per_blob, 2)) for cx, cy in centers]
    pts.append(rng.uniform(0, 20000, size=(noise, 2)))      # sparse background
    P = np.vstack(pts)
    inp = os.path.join(tmp, "locs.csv")
    pd.DataFrame({"frame": rng.integers(1, 1000, len(P)),
                  "x": P[:, 0], "y": P[:, 1]}).to_csv(inp, index=False)
    return inp


def test_srtesseler_finds_blobs_and_noise():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make(tmp)
        out = os.path.join(tmp, "clusters.csv")
        run(input_csv=inp, output_csv=out,
            params={"density_factor": 2.0, "min_cluster_size": 20})
        c = pd.read_csv(out)
        n_clusters = len({int(v) for v in c["cluster_id"] if v >= 0})
        assert n_clusters == 3, f"expected 3 dense blobs, got {n_clusters}"
        assert (c["cluster_id"] < 0).sum() > 0, "sparse background should be noise (-1)"
        assert os.path.exists(os.path.join(tmp, "cluster_summary.csv"))


if __name__ == "__main__":
    test_srtesseler_finds_blobs_and_noise()
    print("cluster_voronoi (SR-Tesseler) test passed")
