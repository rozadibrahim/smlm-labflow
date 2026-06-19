# Environments & isolation

**Rule: isolate on conflict, not by default.** Light tools that are
dependency-compatible live in the labflow core env (fast, in-process). Heavy or
conflicting tools (torch / TensorFlow / CUDA / Java / MATLAB) each get their own
environment. A method's `runtime` in `config/methods.yaml` is the per-tool
isolation decision: `python` = core env; `venv`/`conda`/`docker` = isolated.

## 1. The labflow core env — one command, any machine

The project carries its own setup. On a laptop, a CI runner, or a rented GPU pod,
the same command materializes the isolated core env:

```bash
python bootstrap.py                 # create envs/labflow, install the light core
python bootstrap.py --extras light,gui
python bootstrap.py --from-lock     # reproduce EXACTLY from this platform's lockfile
python bootstrap.py --lock          # (re)generate this platform's lockfile
python bootstrap.py --dry-run       # print the commands, change nothing
```

It's pure standard library and OS-aware (handles `Scripts` vs `bin`), so nothing
is machine-specific. `[light]` = the in-process backends compatible with the core
(scikit-learn, trackpy, scipy, locan, xgboost, lightgbm), which run `drift`,
`cluster`, `spatial_stats`, and `track` in-process. `[gui]` adds napari (needs a
display); `[dev]` adds pytest.

**Python version.** labflow core needs **Python ≥3.11** (snakemake 8). bootstrap
checks the interpreter and tells you how to get one if it's older — point it at a
3.11 with `--python`:

```bash
python bootstrap.py --python python3.11
```

**Lockfiles are per-platform.** A `pip freeze` on Windows is not installable on
Linux, so locks are tagged: `requirements/core.<linux|macos|win>.lock.txt`.
Generate each on its own OS with `--lock`; reproduce with `--from-lock`. `envs/`
is git-ignored — recreate from the lockfile anywhere.

### On a RunPod (or any GPU) pod
The pod is already a CUDA container, so labflow core gets its **own** env there
(the pod's torch env is usually Python 3.10 — too old for snakemake 8, and you
don't want to perturb torch). GPU tools then use the pod's CUDA/torch or their own
containers; conflicting tools get a venv/conda.

```bash
git clone <repo> && cd smlm-labflow
python3.11 -m venv envs/labflow && python bootstrap.py --python python3.11
source envs/labflow/bin/activate
labflow doctor                      # confirms: isolated env, engine, tool placement
python bootstrap.py --lock          # commit requirements/core.linux.lock.txt for the lab
```

`docker/Dockerfile` (pytorch 2.2.2 / cuda 12.1 + the SMLM stack) is the heavy
image and a ready **pod base**: build/push it and select it as the RunPod template,
then bootstrap labflow core on top. Note: a standard pod often can't run *nested*
Docker, so per-tool isolation there is venv/conda, not the per-tool images (those
are for workstations / HPC-with-docker / Apptainer).

## 2. `labflow install <tool>` — pull prebuilt, isolated, reproducible

A lab installs the light core, then *asks* for a tool. For heavy tools the
default is **pull, don't build**: labflow downloads a version-pinned image that
was built once by CI (see §5). Nothing is *solved* on the user's machine, so the
whole class of "it won't install on my laptop" dependency errors disappears — the
biologist gets the exact bits that passed CI.

```bash
labflow install cellpose          # docker:  pull ghcr.io/<owner>/smlm-cellpose
labflow install cellpose --build  # opt out: build docker/cellpose locally instead
labflow install miro              # venv:    envs/miro + git clone + pinned reqs
labflow install ripley            # extra:   pip install ".[stats]" into the core env
labflow install <tool> --dry-run  # print the exact plan, install nothing
labflow installed                 # show what's present vs  -> labflow install <x>
```

The install kind is derived from `runtime` + `install:`:

| runtime | install spec | what runs (default) |
|---|---|---|
| `docker` | `{pull: true}` | `docker pull <image>`  (the one-liner) |
| `docker` | `{context: docker/<tool>}` + `--build` | `docker build -t <image> docker/<tool>` |
| `venv` | `{git:…, requirements_in_src:…, pip:[…], editable:true}` | `python -m venv envs/<env>` + clone + pip |
| `conda` | `{conda_file: envs/specs/<tool>.yml}` | `conda env create -n <env> -f …` |
| `python` | `{extra: stats, probe: <module>}` | `pip install ".[stats]"` (core env) |

**Container engine** (set `LABFLOW_CONTAINER_ENGINE`, default `docker`):

```bash
LABFLOW_CONTAINER_ENGINE=apptainer labflow install cellpose
#   -> apptainer pull envs/sif/cellpose.sif docker://<image>   (daemonless / HPC / no root)
```

Both `install` and `run` honour the engine, so the same method runs under Docker
on a workstation or Apptainer on a cluster with no config change.

**GPU.** A method with `gpu: true` is run with `--gpus all` (Docker) or `--nv`
(Apptainer) automatically. The image ships the CUDA libraries; the host still
needs an NVIDIA driver + `nvidia-container-toolkit` (Docker) for the GPU to be
visible. Without a GPU the same image runs on CPU.

Once installed, the tool just runs (`labflow run <stage> -b <tool>`) — the run
gate checks the env/container/import exists, not a static flag. The recipes below
are exactly what `labflow install` automates (and what you fill in when adding a
tool's `install:` block).

## 3. How tools stay conflict-free

Placement (above) is enforced by four mechanisms, so any two tools — even with
incompatible dependencies — run in one pipeline without colliding:

1. **One process per tool, files between them.** Tools never share a Python import
   namespace; the only thing crossing a boundary is a file (CSV / TIFF) via a
   subprocess. The `runtime` field picks how that process is spawned. Two tools on
   different torch/CUDA versions can't conflict because they're never loaded
   together — this is the core guarantee; the rest make it trustworthy.
2. **Contract validation** (`labflow/contract.py`). Every run's output is checked
   against the columns the next stage reads, so a mislabelled column fails at the
   boundary with a clear message instead of crashing deep inside the next tool.
   Override with `LABFLOW_NO_VALIDATE=1`.
3. **Provenance manifest** (`<output>.labflow.json`, `labflow/provenance.py`). Each
   output records method, isolation level, image digest / env, params, and the
   input+output hashes — so a result is reproducible and auditable, not just
   repeatable. Override with `LABFLOW_NO_PROVENANCE=1`.
4. **Core-env guard.** `labflow install` refuses to install a pip-extra into a
   global interpreter (override: `LABFLOW_ALLOW_GLOBAL=1`) — keeping the core env
   clean is what keeps the in-core tools mutually compatible.

`labflow doctor` shows the whole picture: whether you're in an isolated core env,
the active container engine, and every tool's isolation level.

## 4. Per-tool specs (what `labflow install` runs)

One env per tool (or per conflict-cluster). Each is referenced by a method's
`env:` / `image:` + `install:` in `config/methods.yaml`.

### Localizers (DL, GPU) — `localize`
```bash
# DECODE
conda create -n decode python=3.9 -y && conda activate decode
pip install decode-fish   # or: git clone https://github.com/TuragaLab/DECODE && pip install -e DECODE
# FD-DeepLoc
git clone https://github.com/Li-Lab-SUSTech/FD-DeepLoc && conda env create -f FD-DeepLoc/environment.yml
```

### Segmentation (DL) — `segment`
```bash
# Cellpose via a prebuilt image — the end user pulls, never builds.
labflow install cellpose                                   # docker pull ghcr.io/<owner>/smlm-cellpose
labflow run segment -b cellpose -i image.tif -o masks.tif --param diameter=30
# (image in -> mask label TIFF out; the method is runtime: docker in methods.yaml)
# To rebuild the image from source (docker/cellpose/Dockerfile + contract runner):
labflow install cellpose --build

# StarDist (TensorFlow) — its own env:
python -m venv envs/stardist && envs/stardist/Scripts/pip install stardist tensorflow
```

Why pull, not build: building runs the same fragile installers on every machine,
so any flake (a yanked wheel, a base-image change, a missing system lib) becomes
the lab's problem. Building **once in CI** and pulling a frozen, digest-pinned
image gives every lab the exact bits that passed tests — full isolation, no host
pollution, no solver. `--build` stays as the escape hatch; `mamba` (fast solver,
no daemon) is the lighter alternative when a tool doesn't need a container.

### Tracking — `track`
```bash
# trackpy is in the core (light). MAGIK / TrackMate / Spot-On are isolated:
git clone https://github.com/DeepTrackAI/DeepTrack2          # MAGIK (deeplay/torch)
#   build smlm-labflow/magik image, or a conda env, per its README
#   TrackMate: install Fiji (https://fiji.sc), drive headless via scripting
python -m venv envs/spot_on && envs/spot_on/Scripts/pip install Spot-On-cli
```

### Cluster (DL) — `cluster`
```bash
git clone https://github.com/DeepTrackAI/MIRO
python -m venv envs/miro && envs/miro/Scripts/pip install -r MIRO/requirements.txt   # deeplay/torch
#   then bind labflow/stages/miro_run.py to MIRO's model API + provide a trained model
```

### Phenotype / analyze (DL) — `phenotype`, `analyze`
```bash
# ClusterNet: own conda env (torch). DeepTRACE: build smlm-labflow/deeptrace image
git clone https://github.com/<deeptrace-repo> && docker build -t smlm-labflow/deeptrace docker/deeptrace
```

### QC audit (ML) — `qc_audit`
```bash
# xgboost/lightgbm are in the core (light). For a separate env:
python -m venv envs/qc_ml && envs/qc_ml/Scripts/pip install xgboost lightgbm scikit-learn
```

## 3. Verifying a tool is connected
Once its env exists, flip the method `status: ready` and:
```bash
labflow run <stage> -b <tool> -i in.csv -o out.csv      # headless
labflow run <stage> -b <tool> --gui -i in.csv           # its GUI (if any)
```
A `planned` method gives a clear "build its env/adapter" message until then.

**Conformance — prove what actually runs on this machine.** `labflow conformance`
smoke-tests *every* registered method end-to-end on a synthetic fixture for its
stage, and reports per tool:

```bash
labflow conformance                 # PASS / FAIL / SKIP for every method
labflow conformance --stage cluster
```

- **PASS** – the tool ran and produced a contract-valid output here.
- **SKIP** – not installed / incomplete scaffold / needs real data — not a failure.
- **FAIL** – an installed, ready tool that did not produce a valid output.

It exits non-zero on any FAIL, so it is also the CI regression gate
(`labflow/tests/test_conformance.py` asserts the in-core tools pass). This is the
preflight a lab runs after `labflow install …` to confirm the toolset works before
trusting a real run.

## 5. Publishing the prebuilt images (maintainer)
The images labs pull are built once by CI from `docker/<tool>/` and pushed to
GHCR — `.github/workflows/build-images.yml`:

```bash
git tag v0.3.0 && git push --tags     # builds every docker/<tool>/ and pushes :latest + :v0.3.0
#   or run the workflow manually (Actions -> build-tool-images) for one tool
```

To add a tool to the fleet: drop a `docker/<tool>/Dockerfile` (+ contract runner),
add `<tool>` to the workflow `matrix.tool`, and point the method's `image:` at
`ghcr.io/<owner>/smlm-<tool>` with `install: {pull: true, context: docker/<tool>}`.
For bit-exact reproducibility, pin `image:` to the published `@sha256:…` digest.

## 6. For a distributable standard
Pin every env (lockfiles) and containerize per heavy step (`runtime: docker`,
pulled by digest), so a collaborator reproduces results identically. The
`runtime` field lets each method opt into a container without touching anything
else; light methods stay in-process in the core env.
