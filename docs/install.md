# Installing SMLM LabFlow

The supported engineering path is currently a **source checkout on Linux with
Python 3.12**, followed by an isolated core and optional backend environments.
A registry entry is not a promise that its adapter is implemented. See
[engineering acceptance](engineering-acceptance.md) for the measured status.

```bash
git clone --branch codex/runpod-pipeline-image https://github.com/rozadibrahim/smlm-labflow.git
cd smlm-labflow
python3.12 bootstrap.py --extras light,dev
source envs/labflow/bin/activate
labflow doctor
labflow demo --out outputs/demo
labflow conformance --output-dir outputs/conformance --require rcc --require trackpy
```

The analysis core requires Python >=3.11. The LiteLoc backend has its own Python
3.9/spline environment; do not install the analysis core into it.

## Optional backends

```bash
labflow install cellpose
labflow install stardist
labflow install microsam
labflow install omnipose
labflow installed
labflow conformance --stage segment --require cellpose --require stardist \
  --output-dir outputs/segmentation-check
labflow run segment -b cellpose -i image.tif -o masks.tif
```

These four segmenters install into separate virtual environments. Their default
recipes do not need Docker. Cellpose, micro-SAM and Omnipose use pinned CUDA 12.8
PyTorch wheels compatible with the tested Blackwell Pod; StarDist uses isolated
CPU TensorFlow. GPU use is a backend parameter, not evidence that a model was
validated on your microscope data. Models may download on first execution.

`labflow install liteloc` installs a pinned upstream source checkout and checks
its imports. On the supplied RunPod image it reuses `LABFLOW_LEGACY_PYTHON`.
Without that interpreter, its Linux recipe requires conda and creates a separate
Python 3.9 environment. The existing-image reuse path has been tested on the Pod;
a completely fresh LiteLoc environment from this new command still needs its own
clean-install CI gate. Run `scripts/check_liteloc_engineering.py` with that backend
interpreter to test small CPU/CUDA operations on the upstream network. Models,
PSF calibration and experiment profiles are separate inputs.

Unimplemented adapters fail explicitly. Installing packages cannot complete an
adapter. Do not treat SKIP as engineering acceptance: `--require NAME` demands an
actual PASS, including when the named method was not selected by `--stage`.

## RunPod storage

Keep source, data, model downloads and validation results on `/workspace`.
Keep executable environments on the container's local filesystem when the
network volume cannot preserve executable/private-file permissions:

```bash
export LABFLOW_ENV_ROOT=/opt/labflow-envs
export LABFLOW_MODEL_ROOT=/workspace/models
```

Set these consistently for both installation and execution. They override the
portable defaults `envs/` and `models/` beneath the checkout. Container replacement
loses environments under `/opt`; the recipes and recorded package lists make
reinstallation possible. They are not automatically installed during SSH startup.

## Distribution and GUI status

Conda-forge/PyPI publication, a verified multiplatform package, offline bundles,
and a complete GUI workflow are separate release tasks. This document does not
claim those distribution routes are already published or tested. Napari review
is optional and needs a suitable display; headless backend execution is tested
separately from desktop GUI behavior.
