# RunPod image for the expanded pipeline

Build the `runpod` target from the `feat/labflow-pipeline` branch (or its image
update branch). The `liteloc` target preserves the previous batch image.

```bash
docker build --target runpod -f docker/Dockerfile -t smlm-labflow:runpod .
bash scripts/runpod_smoke_test.sh smlm-labflow:runpod
```

The **Build RunPod pipeline image** GitHub Actions workflow builds, checks the
scientific core, tests a real SSH login with a mounted volume, and only then
publishes `ghcr.io/<owner>/smlm-labflow:runpod-<full-commit-sha>`. Copy the exact
image reference from the successful workflow summary. It does not replace
`:latest`. Private GHCR packages need registry credentials in RunPod.

## Template settings

- Image: the tested `runpod-<full-commit-sha>` tag from the workflow summary.
- TCP port: `22`.
- Volume mount: `/workspace`; attach persistent storage for data and results.
- Environment variable `PUBLIC_KEY`: your full SSH **public** key. Check that
  RunPod has populated it from your account key, or paste the public key here.
- Leave container entrypoint/start command unset so the image starts SSH.

No private keys or credentials belong in the image. Host keys are generated on
first boot and kept under `/workspace/.labflow-ssh` for that volume. Avoid sharing
one volume's SSH host identity between independently exposed pods.

The template provides SSH/SFTP, rsync, tmux and SSH port forwarding. It does not
start Jupyter or a desktop GUI. Review napari outputs on your local workstation.

## Included environments

| Command | Environment and purpose |
|---|---|
| `labflow`, `python` | Isolated Python 3.12 core, `[light]` analysis dependencies, Snakemake 8 |
| `labflow-liteloc` | Legacy `run_pipeline.py` using isolated Python 3.9, Torch 2.2.2/CUDA 12.1 and spline |
| `labflow run localize -b liteloc ...` | Registry explicitly selects the legacy Python interpreter |

The newer core does not upgrade LiteLoc's Torch or spline ABI. Do not assume the
legacy CUDA stack supports a newer GPU architecture; validate on your target GPU
before running experiments. The build and CI checks do not use a GPU.

The image contains the LiteLoc **dependencies and LabFlow adapter**, not the
external LiteLoc source, trained models, or microscope calibration. Install your
compatible LiteLoc checkout at `/workspace/backends/LiteLoc`, or supply its path
through your profile. Use `labflow-liteloc calibrate/train/infer ...` for the legacy
workflow. Calibration and inference require your real inputs and profiles.

```bash
labflow doctor
labflow conformance
labflow demo --out /workspace/outputs/demo
/opt/conda/envs/smlm-labflow/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Conformance reports PASS / FAIL / SKIP; a registered method is not necessarily
installed or implemented. Heavy optional tools (Cellpose, StarDist, MAGIK, etc.)
retain their own environments. Standard RunPod pods cannot be assumed to support
nested Docker: their existing `runtime: docker` entries need a supported separate
execution host or an explicitly implemented venv/conda setup. Nextflow, Julia,
Fiji and MATLAB are not bundled in this image.

Code and baked environments live in `/opt`, so mounting `/workspace` cannot hide
them. Store inputs, outputs, models and backend sources under `/workspace`.
Per-tool `envs/` installations point into `/workspace/envs`; named conda envs
created elsewhere still need their own persistence configuration. The resolved
core package list is recorded at `/opt/labflow-core.freeze.txt` in each image;
use the published image digest to reproduce the exact built environment.
