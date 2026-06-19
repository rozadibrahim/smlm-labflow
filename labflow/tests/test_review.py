"""
Test the GUI-free part of the napari review bridge (gather_layers).
The actual napari window needs a display and isn't exercised here.

Run: python labflow/tests/test_review.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.review import gather_layers


def test_gather_layers_finds_locs_and_clusters():
    with tempfile.TemporaryDirectory() as tmp:
        drift = os.path.join(tmp, "batches", "b1", "drift")
        os.makedirs(drift)
        pd.DataFrame({"frame": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}).to_csv(
            os.path.join(drift, "drift_corrected_localizations.csv"), index=False)

        clu = os.path.join(tmp, "batches", "b1", "cluster")
        os.makedirs(clu)
        pd.DataFrame({"frame": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0],
                      "cluster_id": [0, 0, -1]}).to_csv(
            os.path.join(clu, "clusters.csv"), index=False)

        layers = gather_layers(tmp)
        names = [L["name"] for L in layers]
        assert any("localizations" in n for n in names), names
        clusters = [L for L in layers if L["name"] == "clusters"]
        assert clusters and clusters[0]["data"].shape == (3, 2)
        assert clusters[0]["cluster_id"] is not None


if __name__ == "__main__":
    test_gather_layers_finds_locs_and_clusters()
    print("review (gather_layers) test passed")
