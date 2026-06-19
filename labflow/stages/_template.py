"""
labflow.stages._template

Copy-paste starting point for a NEW in-process (light, dependency-compatible)
method. Heavy or conflicting tools (torch / TensorFlow / Java / MATLAB) do NOT go
here -- they run isolated as a subprocess; see docs/methods.md and docs/environments.md.

To plug in a light in-process method:

  1. Copy this file to  labflow/stages/<your_stage>.py  and write the science in
     `run()`. Use the spine's IO (read_localizations / read_table / write_table)
     so files stay on the canonical contract -- never re-implement CSV handling.

  2. Register ONE entry in config/methods.yaml:

        my_method:
          stage: cluster                       # an existing stage (see `labflow list`)
          runtime: python
          entry: "labflow.stages.your_stage:run"
          output: clusters.csv                 # the contract filename it produces
          params: {eps: 100, min_samples: 10}  # defaults, overridable with --param
          status: ready
          install: {extra: cluster, probe: sklearn}   # omit if numpy/pandas-only
          description: "One line shown by `labflow list`."

     That single entry makes it available in BOTH `labflow run <stage> -b my_method`
     and the Snakemake pipeline. `labflow doctor` will lint the entry.

  3. For a FAMILY of methods that share one adapter (e.g. dbscan/optics/hdbscan),
     give each registry entry the same `entry:` and a `bind:` that selects the
     variant, then branch on it inside `run()` (see how `algorithm`/`metric`/
     `backend` are used in cluster.py / spatial_stats.py / drift.py).

The runner calls `run()` by keyword; keep this exact signature.
"""

from __future__ import annotations

from typing import Any, Dict

from ..io import read_localizations, write_table


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    """Consume the previous stage's CSV, produce this stage's CSV.

    input_csv  : path to the upstream canonical CSV (localizations / clusters / ...)
    output_csv : where to write this stage's output (the runner validates it
                 against the file contract and records provenance afterwards)
    params     : merged defaults + bind + CLI --param; pop control keys you don't use
    returns    : the output path (always `return write_table(df, output_csv)`)
    """
    p = dict(params or {})
    p.pop("pixel_size_nm", None)        # present for every stage; ignore if unused
    p.pop("units", None)

    locs = read_localizations(input_csv)   # or read_table() for non-localization input

    # --- the science: compute your result into a DataFrame -------------------
    out = locs.copy()
    out["example_metric"] = 0.0

    # One line out; write_table creates parent dirs and returns the path.
    return write_table(out, output_csv)
