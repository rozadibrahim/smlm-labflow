# RunPod image for the expanded pipeline

Build the `runpod` target from the `feat/labflow-pipeline` branch (or its image
update branch). The `liteloc` target preserves the previous batch image.

For the fastest start on an existing PyTorch pod, use the Conda recipe instead
of rebuilding CUDA layers:

```bash
conda env create -f env_yamls/labflow_core_linux.yml
conda run -n labflow-core python -m pip install --no-deps -e .
conda run -n labflow-core labflow doctor
conda run -n labflow-core labflow conformance
```

This supplies the analysis core, not LiteLoc or other tools' conflicting GPU
dependencies. For a reusable pod with SSH and the legacy dependency environment,
build the Docker target below. Dependency layers precede source copies, so code
edits reuse the expensive environments.

```bash
docker build --target runpod -f docker/Dockerfile -t smlm-labflow:runpod .
bash scripts/runpod_smoke_test.sh smlm-labflow:runpod
```

The **Build RunPod pipeline image** GitHub Actions workflow builds, checks the
scientific core, tests a real SSH login with a mounted volume, and only then
publishes `ghcr.io/<owner>/smlm-labflow:runpod-<full-commit-sha>`. Copy the exact
image reference from the successful workflow summary. The convenient `:runpod-feat`
alias points at the most recently tested development build; use the commit tag
or digest to reproduce a particular build. It does not replace `:latest`.
Private GHCR packages need registry credentials in RunPod.

Heavyweight CI builds are opt-in: create or update `docker/runpod/build-request.txt`
and push it to the development branch to request a build of that commit. Ordinary
source pushes only run the fast tests. The workflow also supports manual dispatch
where GitHub exposes it. A local build/push is available at any time:

```bash
docker build --target runpod -f docker/Dockerfile -t ghcr.io/<owner>/smlm-labflow:runpod-feat .
bash scripts/runpod_smoke_test.sh ghcr.io/<owner>/smlm-labflow:runpod-feat
docker push ghcr.io/<owner>/smlm-labflow:runpod-feat
```

## Template settings

- Image: the tested `runpod-<full-commit-sha>` tag from the workflow summary.
- TCP port: `22`.
- Volume mount: `/workspace`; attach persistent storage for data and results.
- Environment variable `PUBLIC_KEY`: your full SSH **public** key. Check that
  RunPod has populated it from your account key, or paste the public key here.
- Leave container entrypoint/start command unset so the image starts SSH.

### Startup compatibility candidate

The `docker/runpod/Dockerfile.startup` overlay preserves the published scientific
environment at commit `64263d3` and changes only startup. It clears the custom
ENTRYPOINT and provides a directly executable `/start.sh` as CMD. The full build
uses the same startup convention. If the template overrides the start command,
set it explicitly to `/start.sh` and leave the entrypoint unset.

The **Test and publish RunPod startup candidate** workflow checks default startup,
an injected `/bin/bash -c 'exec /start.sh'` command, SSH, the synthetic demo and
storage persistence and correct host-key lifecycle before publishing `:runpod-startup-fix` and a
commit-specific `:runpod-startup-<sha>`. It does not update `:runpod-feat` or `:latest`.
This small overlay avoids reinstalling the CUDA and scientific dependencies.

The first application log is `LabFlow startup: entering /start.sh`; shell failures
report a line number and exit status without logging commands or credentials.
This candidate addresses startup-command compatibility and diagnostics. It is
not a confirmed fix for RunPod's `error starting sidecar ... runc ... EOF` failure,
which may occur before any image startup code runs. Real RunPod validation remains
required even when the Docker integration tests pass.

No private keys or credentials are baked into the image. Host keys are generated
on first boot under `/etc/ssh/labflow`, on the container's local filesystem. Some
RunPod network volumes report files as mode `0666` even after `chmod 600`; OpenSSH
refuses private keys on those volumes. Earlier images' `/workspace/.labflow-ssh`
keys are ignored, not imported. Your public login key is still installed under
`/root/.ssh`; the private login key stays on your own computer.

An ordinary Docker restart preserves the container's host key. Replacing or
resetting the container's local disk creates a new host identity, even when the
same network volume is attached. If SSH reports a changed identity after a known
replacement, verify its new fingerprint through RunPod's console before updating
that host's known_hosts entry. Do not disable host-key verification globally.
Scientific datasets and results remain under `/workspace`.

Startup does not install packages or launch scientific jobs. A workspace-directory
creation failure is reported without preventing SSH access for diagnosis. The
SSH configuration and host key are checked before printing the readiness message.

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

### Blackwell GPUs on the current image

The RTX PRO 6000 Blackwell (sm_120) cannot execute kernels from the baked
Torch 2.2.2/CUDA 12.1 installation. After SSH login, use the optional upgrade
overlay from this branch:

```bash
bash scripts/upgrade_runpod_blackwell.sh
```

Run this from a checkout containing that script and `requirements/blackwell.txt`.
It backs up the existing GPU environment under `/opt`, then installs the official
Torch 2.7.1, torchvision 0.22.1 and torchaudio 2.7.1 CUDA 12.8 wheels into the
existing Python 3.9 environment. NumPy 1.24.4 and the conda spline package stay
in place, so the existing CLI, registry and workflow interpreter paths still work.
This is an explicit repair step, not part of SSH startup. Version pairings follow
the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v271).

The repair records before/after package lists and dependency checks beneath
`/workspace/outputs/blackwell-upgrade-*`. The legacy environment already reports
missing core-only click/Snakemake dependencies and hdfdict's stale PyYAML bound;
the script requires the upgrade to introduce no change to those diagnostics.
It checks scientific imports and runs CUDA arithmetic, matrix multiplication,
convolution forward/backward, an optimizer step and torchvision's CUDA NMS.
These checks establish runtime compatibility, not LiteLoc model accuracy or
validation of the optional toolkit backends.

The live environment and rollback copy are container-local. A replacement image
starts from its baked packages, so keep the repair files on `/workspace` and
reapply explicitly if that local disk is reset. Updating a running Pod does not
update the GHCR image. `backup-path.txt` in the repair record identifies the
complete old environment: to roll back, stop GPU jobs, move the modified
`/opt/conda/envs/smlm-labflow` aside, then move that backup back to the exact
original path. Do not run the backup at its temporary prefix.

`LABFLOW_LEGACY_PYTHON` routes the Snakemake/Nextflow localization lifecycle to
the legacy interpreter too. The baked Snakemake defaults read inputs and write
results beneath `/workspace`; override the scientific settings with your profile.

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
