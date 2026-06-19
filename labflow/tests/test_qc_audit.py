"""Unsupervised QC audit (labflow.stages.qc_audit)."""

import numpy as np
import pandas as pd

from labflow.stages.qc_audit import run


def test_flags_injected_outliers(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    loc = pd.DataFrame({"photons": rng.normal(5000, 500, n),
                        "confidence": rng.uniform(0.8, 1.0, n)})
    loc.loc[:9, "photons"] = 50          # 10 clearly-bad rows
    loc.loc[:9, "confidence"] = 0.1
    inp = tmp_path / "locs.csv"
    loc.to_csv(inp, index=False)
    out = tmp_path / "qc.csv"

    run(input_csv=str(inp), output_csv=str(out), params={"contamination": 0.05})
    qc = pd.read_csv(out)

    assert {"qc_score", "qc_flag"} <= set(qc.columns)
    assert (qc.loc[:9, "qc_flag"] == "fail").sum() >= 7      # most outliers caught
    assert qc["qc_score"].between(0.0, 1.0).all()
