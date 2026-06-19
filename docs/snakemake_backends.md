# Snakemake workflow & pluggable backends

A thin [Snakemake](https://snakemake.github.io/) wrapper orchestrates the
existing pipeline and adds a **pluggable drift-correction backend** that
replaces the old centroid drift *proxy* with real drift estimation. Nothing in
`run_pipeline.py` changed — Snakemake calls it as-is.

## DAG

```
calibrate ─▶ train ─▶ infer ─▶ drift_correct ─▶ report
(optional)   (opt.)  checkpt    pluggable        per run
```

`calibrate`, `train`, `infer` all reuse `run_pipeline.py` and nest under one
`run_dir`, sharing `run_dir/registry` — which is how the pipeline already couples
them: calibrate writes `latest_calibration.json`, train reads it and writes
`latest_model.json`, infer reads that. The calibrate/train rules are **gated by
config flags** so a normal run does not retrain: `infer` reuses the model already
in the registry.

`infer` owns its own per-movie batch-folder naming, so it is a Snakemake
*checkpoint*: the drift and report rules discover the batches it produced from
`batch_manifest.csv` and fan out automatically.

## Run

```bash
pip install -r requirements/workflow.txt

# edit config/config.yaml (input_dir, profile, backend, drift_backend), then:
snakemake --cores 4                                  # infer -> drift -> report
snakemake --cores 4 --config train=true calibrate=true   # full LiteLoc lifecycle
snakemake --cores 4 --config drift_backend=dme       # override any config key
snakemake --cores 4 -n                               # dry-run, print the DAG

# or run a lifecycle stage on its own:
snakemake --cores 4 calibrate
snakemake --cores 4 train
```

The localization backend is `backend:` in config (default `liteloc`, passed to
`run_pipeline.py -b`). A fresh `run_dir` has an empty registry, so either run the
lifecycle once (`--config train=true calibrate=true`) or drop an existing
`latest_model.json` into `run_dir/registry/`.

Outputs per batch land in
`results/batches/<id>/drift/`:
`drift_corrected_localizations.csv`, `drift_trajectory.csv`,
`drift_correction.json`.

## Drift-correction backends (shipped)

| name   | method                                       | notes |
|--------|----------------------------------------------|-------|
| `none` | passthrough (no correction)                  | keeps the legacy proxy behaviour |
| `rcc`  | redundant cross-correlation (Wang 2014)      | pure numpy, always available |
| `aim_julia` | adaptive intersection maximization (Ma et al. 2024) | **faithful Julia port of the authors' MATLAB**; needs a Julia runtime |
| `dme`  | drift at minimum entropy (Cnossen 2021)      | adapter to the upstream `dme` package |

`rcc` recovers an injected drift curve to ~2–3 nm RMS and runs on ~650 k
localizations in a few seconds with no exotic deps. **`aim_julia` is the most
accurate** — a near-line-by-line translation of `AIM.m`/`IntersectionMax.m`
(`driftcorr/julia/aim.jl`, stdlib-only), and it recovers the injected drift to
**sub-nm** (~0.3 nm). It needs a Julia runtime (`winget install Julialang.Juliaup`)
but no Julia packages. `dme` (entropy minimization) needs the upstream package and
ideally CUDA.

Provenance, to be precise: `aim_julia` translates the authors' actual MATLAB
(https://github.com/YangLiuLab/AIM — no explicit license; cite Ma et al., Sci.
Adv. 2024); `dme` calls the authors' real Python package.

Tune via `drift_params` in `config/config.yaml` — e.g. `rcc`: `n_time_bins`,
`render_nm`, `neighbor_span`, `max_drift_nm`; `aim_julia`: `intersect_nm`,
`track_interval`, `julia_bin`.

## Add a drift backend (the extension point)

1. Write a function in a new `driftcorr/<name>.py`:

   ```python
   def estimate_drift_xxx(locs, *, pixel_size_nm, units, params):
       # locs: DataFrame with frame, x, y[, z, lpx, lpy]
       # return a driftcorr.core.DriftEstimate (per-frame dx, dy, dz)
       ...
   ```

2. Register it in `driftcorr/registry.py`:

   ```python
   DRIFT_BACKENDS["xxx"] = estimate_drift_xxx
   ```

3. Select it: `drift_backend: xxx` in `config/config.yaml` (add a
   `drift_params: { xxx: {...} }` block if it takes parameters).

That's it — the CLI (`python -m driftcorr.run`), the Snakemake rule, and the
report are all backend-agnostic and need no changes. `driftcorr/tests/test_rcc.py`
shows how to test one against an injected drift curve.

## Add a localization backend

The localization seam already exists in the repo: every backend converges on
`canonical_localizations.csv` (`schema.py`), which is what `drift_correct` and
everything downstream consume.

1. Add the machine wiring under a new top-level key in
   `adapters/backend_paths.yml` (modules / functions / execution), mirroring the
   `liteloc:` block.
2. Implement the adapter the resolver dispatches to (see
   `adapters/liteloc_adapter.py` and `adapters/resolver.py`).
3. Select it with `backend: <name>` in `config/config.yaml` (or
   `run_pipeline.py infer -b <name>`).

Downstream Snakemake rules are unchanged because they only depend on the
canonical CSV contract.
