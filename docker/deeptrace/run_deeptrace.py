#!/usr/bin/env python
"""
DeepTRACE track-analysis runner -- runs INSIDE the smlm-labflow/deeptrace image (DL,
2026), invoked by labflow's `runtime: docker` analyze method.

Contract: tracks.csv (track_id, frame, x, y) in -> track_analysis.csv (track_id +
per-track properties) out, the labflow `analyze` stage output.

BINDING POINT: DeepTRACE is a 2026 deep-learning track analyser whose API/model are
not pinned here. The IO + the analyze contract are complete -- wire `_analyze()` to
the model in your image. It raises a clear message until then (no fabricated API).
"""

import argparse
import json

import pandas as pd


def _analyze(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return one row per track including `track_id`. DeepTRACE binding point."""
    raise NotImplementedError(
        "DeepTRACE inference is the binding point: wire docker/deeptrace/run_deeptrace.py:"
        "_analyze to the model. IO + the track_analysis contract are done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    result = _analyze(pd.read_csv(args.inp), json.loads(args.params))
    result.to_csv(args.out, index=False)
    print(f"deeptrace: {len(result)} tracks analysed -> {args.out}")


if __name__ == "__main__":
    main()
