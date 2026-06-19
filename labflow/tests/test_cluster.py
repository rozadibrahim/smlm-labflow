"""
Regression test for the cluster stage (sklearn backends).

Run: python -m pytest labflow/tests/test_cluster.py
(or:  python labflow/tests/test_cluster.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.stages.cluster import run


def _make_blobs(tmp, n_blobs=3, per_blob=300, noise=200):
    rng = np.random.default_rng(0)
    centers = [(5000, 5000), (15000, 5000), (10000, 15000)][:n_blobs]
    pts = [rng.normal((cx, cy), 80, size=(per_blob, 2)) for cx, cy in centers]
    pts.append(rng.uniform(0, 20000, size=(noise, 2)))   # scattered noise
    P = np.vstack(pts)
    frame = rng.integers(1, 1000, size=len(P))
    inp = os.path.join(tmp, "locs.csv")
    pd.DataFrame({"frame": frame, "x": P[:, 0], "y": P[:, 1]}).to_csv(inp, index=False)
    return inp


def _n_clusters(out_csv):
    c = pd.read_csv(out_csv)
    return len(set(int(v) for v in c["cluster_id"] if v >= 0))


def test_dbscan_finds_three_blobs():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_blobs(tmp)
        out = os.path.join(tmp, "clusters.csv")
        run(input_csv=inp, output_csv=out,
            params={"algorithm": "dbscan", "eps": 300, "min_samples": 15})
        assert _n_clusters(out) == 3
        assert os.path.exists(os.path.join(tmp, "cluster_summary.csv"))


def test_hdbscan_if_available():
    try:
        from sklearn.cluster import HDBSCAN  # noqa: F401
    except ImportError:
        print("SKIP: scikit-learn<1.3, no HDBSCAN")
        return
    with tempfile.TemporaryDirectory() as tmp:
        inp = _make_blobs(tmp)
        out = os.path.join(tmp, "clusters.csv")
        run(input_csv=inp, output_csv=out,
            params={"algorithm": "hdbscan", "min_cluster_size": 50})
        assert _n_clusters(out) == 3


if __name__ == "__main__":
    test_dbscan_finds_three_blobs()
    test_hdbscan_if_available()
    print("cluster tests passed")
