# Environments and dependency isolation

LabFlow's Python 3.11+ core hosts compatible analysis tools. Heavy backends run
in their own processes and environments; CSV/TIFF artifacts cross the boundary.
The registry in `config/methods.yaml` selects the runtime and installation recipe.
See [installation](install.md) for the tested source-checkout workflow.

## Current runtime paths

| Backend family | Runtime | Installation |
|---|---|---|
| Drift, clustering, tracking, spatial metrics and QC | Core Python | `bootstrap.py --extras light` |
| Cellpose, StarDist, micro-SAM, Omnipose | One venv per tool | `labflow install NAME` |
| LiteLoc | External Python 3.9/spline environment | `labflow install liteloc`; reuse `LABFLOW_LEGACY_PYTHON` on the image |
| AIM Julia port | Julia subprocess | Install Julia on PATH; no extra Julia packages |
| DME | Native upstream library plus Python wrapper | Linux native build remains blocked in the tested setup |
| DECODE | Conda adapter scaffold | Environment recipe and reference inference validation still required |
| Other declared research adapters | Explicit stubs | Implement and validate before marking available |

Container runners remain available for methods with `runtime: docker`. That
mechanism supports Docker or `LABFLOW_CONTAINER_ENGINE=apptainer`; neither engine
is assumed available inside a RunPod container. The four native segmentation
recipes replace their former default Docker requirement. Their Docker build
contexts remain in the repository but are not the tested native installation.

## Installation is separate from acceptance

A venv directory alone does not establish an installation. LabFlow records a
successful recipe receipt only after package installation, `pip check`, and
imports in the **target interpreter** succeed. Changing the declared recipe or
requirements file invalidates its receipt. A failed reinstall removes the old
receipt before mutation, so an incomplete environment does not look ready.

The receipt is `.labflow-install.json`; the resolved package inventory is
`.labflow-freeze.txt` in that environment. These are evidence of a tested install,
not a claim of fully locked cross-platform reproducibility. Top-level backend
versions are pinned; a distribution release still needs complete platform locks.

Execution resolves the same absolute environment path as installation, including
when invoked from another directory or a path containing spaces. Backend Python
runs by its full executable path. The inherited core `PYTHONPATH`/`PYTHONHOME`
are removed for venv commands. Runtime-specific settings and model cache paths
are declared in the registry.

Set `LABFLOW_ENV_ROOT` to choose where environments live and `LABFLOW_MODEL_ROOT`
for persistent models/cache. Defaults are `envs/` and `models/` in the checkout.
On the tested Pod these are `/opt/labflow-envs` and `/workspace/models`; the latter
persists across container replacement while executable environments must be rebuilt.

## Contracts and evidence

LabFlow checks required CSV columns at supported stage boundaries. Segmentation
must produce a readable TIFF containing nonnegative integer labels; labels are
saved as uint32 by the Cellpose/StarDist runners to avoid uint16 overflow.
These are structural checks, not complete validation of units or scientific data.

Each successful run writes `<output>.labflow.json` containing input/output hashes,
parameters and runtime information. Venv outputs also record the selected Python
path and hashes of the installation receipt and package inventory. External
artifacts such as model files need their own provenance review before release.

```bash
labflow conformance --output-dir outputs/acceptance --require rcc --require cellpose
```

This preserves fixtures, per-method outputs, sidecars and a JSON report. Each
method gets its own directory so sidecars cannot overwrite one another.
`--require` fails acceptance on FAIL, SKIP, or an unknown/unselected method.
Supervised XGBoost/LightGBM checks fit tiny disposable models and compare adapter
predictions with direct predictions. LiteLoc's separate engineering script tests
random-weight network execution and serialization, not trained localization.

Fifteen research adapters currently contain explicit unfinished bindings. They
must not advertise a working algorithm merely because dependencies are installed.
