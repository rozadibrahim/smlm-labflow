"""qPAINT molecular counting (labflow.stages.counting)."""

import pandas as pd
import pytest

from labflow.stages.counting import run


def _clusters(path, rows):
    pd.DataFrame(rows, columns=["frame", "x", "y", "cluster_id"]).to_csv(path, index=False)
    return path


def test_qpaint_influx_orders_by_blink_rate(tmp_path):
    # cluster 0 binds every 2 frames (more docking sites), cluster 1 every 5
    rows = [(f, 10, 10, 0) for f in range(0, 21, 2)] + [(f, 50, 50, 1) for f in range(0, 21, 5)]
    out = tmp_path / "counts.csv"
    run(input_csv=str(_clusters(tmp_path / "c.csv", rows)), output_csv=str(out), params={})
    c = pd.read_csv(out).set_index("cluster_id")
    assert c.loc[0, "qpaint_influx"] > c.loc[1, "qpaint_influx"]
    assert -1 not in c.index                          # noise excluded


def test_qpaint_molecules_need_calibration(tmp_path):
    rows = [(f, 10, 10, 0) for f in range(0, 21, 2)]   # influx 0.5
    out = tmp_path / "o.csv"
    run(input_csv=str(_clusters(tmp_path / "c.csv", rows)), output_csv=str(out),
        params={"unit_influx": 0.5})
    assert abs(pd.read_csv(out).loc[0, "n_molecules"] - 1.0) < 1e-6


def test_requires_cluster_id(tmp_path):
    inp = tmp_path / "c.csv"
    pd.DataFrame({"frame": [0], "x": [1], "y": [2]}).to_csv(inp, index=False)
    with pytest.raises(ValueError):
        run(input_csv=str(inp), output_csv=str(tmp_path / "o.csv"), params={})


def test_ibfcs_refuses_cleanly(tmp_path):
    rows = [(0, 1, 2, 0)]
    with pytest.raises(NotImplementedError):
        run(input_csv=str(_clusters(tmp_path / "c.csv", rows)),
            output_csv=str(tmp_path / "o.csv"), params={"method": "ibfcs"})
