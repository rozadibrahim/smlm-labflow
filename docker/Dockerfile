FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS liteloc-deps

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/smlm-labflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY env_yamls/liteloc_env_working_explicit.txt /tmp/liteloc_conda_explicit.txt

RUN echo "===== Creating LiteLoc-compatible Python 3.9 conda environment =====" && \
    conda create -y -n smlm-labflow --file /tmp/liteloc_conda_explicit.txt && \
    conda clean -afy

ENV PATH=/opt/conda/envs/smlm-labflow/bin:$PATH
ENV CONDA_DEFAULT_ENV=smlm-labflow

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements/remote.txt /tmp/remote.txt

RUN echo "===== /tmp/remote.txt =====" && \
    cat /tmp/remote.txt && \
    echo "===== Python / pip versions =====" && \
    python --version && \
    python -m pip --version && \
    echo "===== Installing main Python dependencies =====" && \
    python -m pip install --no-cache-dir --prefer-binary -r /tmp/remote.txt && \
    echo "===== Installing hdfdict without dependency resolver =====" && \
    python -m pip install --no-cache-dir --no-deps hdfdict==0.3.1

RUN echo "===== Patching hdfdict version import =====" && \
    python - <<'PY'
from pathlib import Path
import site

replacement = '''# -- coding: utf-8 --
"""hdfdict helps h5py to dump and load python dictionaries."""

from importlib.metadata import version as _dist_version

from .hdfdict import load, dump

__version__ = _dist_version("hdfdict")
__all__ = ["load", "dump"]
'''

candidates = [
    Path(site_dir) / "hdfdict" / "__init__.py"
    for site_dir in site.getsitepackages()
]

for init_py in candidates:
    if init_py.exists():
        init_py.write_text(replacement, encoding="utf-8")
        print(f"Patched {init_py}")
        break
else:
    raise FileNotFoundError("Could not find hdfdict/__init__.py in site-packages")
PY

RUN echo "===== Import smoke test =====" && \
    python - <<'PY'
import torch
import torchvision
import torchaudio
import yaml
import numpy
import pandas
import scipy
import skimage
import tifffile
import h5py
import h5py_wrapper
import hdfdict
import omegaconf
import natsort
import spline
import thop
import deprecated
import dotted_dict
import recursivenamespace
import psutil
import pynvml

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("PyYAML:", yaml.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-image:", skimage.__version__)
print("h5py:", h5py.__version__)
print("h5py-wrapper import: OK")
print("hdfdict:", hdfdict.__version__)
print("OmegaConf:", omegaconf.__version__)
print("spline import: OK")
print("pynvml import: OK")
print("RecursiveNamespace import: OK")
print("LiteLoc/LabFlow dependency smoke test: OK")
PY

FROM liteloc-deps AS liteloc
COPY . /workspace/smlm-labflow

CMD ["python", "run_pipeline.py", "--help"]

# Interactive RunPod target. Keep the legacy CUDA/spline stack isolated from
# the Python >=3.11 core required by the registry-driven pipeline / Snakemake.
FROM liteloc-deps AS runpod
LABEL org.opencontainers.image.source="https://github.com/rozadibrahim/smlm-labflow"
LABEL org.opencontainers.image.description="LabFlow analysis core and isolated LiteLoc dependencies for interactive RunPod use"

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server rsync tmux tini libgomp1 \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_*

# /workspace is a volume on RunPod; application code and built environments
# must remain visible when an empty or existing volume is mounted there.
WORKDIR /opt/smlm-labflow

RUN conda create -y --override-channels -c conda-forge \
      -p /opt/labflow-python python=3.12 \
    && conda clean -afy \
    && /opt/labflow-python/bin/python -m venv /opt/labflow-core

ENV PATH=/opt/labflow-core/bin:/opt/conda/envs/smlm-labflow/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/smlm-labflow
ENV MPLBACKEND=Agg
ENV LABFLOW_LEGACY_PYTHON=/opt/conda/envs/smlm-labflow/bin/python
ENV CONDA_DEFAULT_ENV=""

# Only dependency metadata invalidates this expensive layer. Editing adapters,
# profiles, docs or application code reuses all built Python/CUDA environments.
COPY pyproject.toml /tmp/labflow-pyproject.toml
RUN python - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path
project = tomllib.loads(Path('/tmp/labflow-pyproject.toml').read_text())['project']
requirements = project['dependencies'] + project['optional-dependencies']['light']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                       'setuptools', 'wheel', *requirements])
PY

COPY . /opt/smlm-labflow
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -m pip check \
    && python docker/runpod/configure.py \
    && python -m pip freeze --exclude-editable > /opt/labflow-core.freeze.txt \
    && labflow doctor \
    && labflow conformance \
    && labflow demo --out /tmp/labflow-demo \
    && rm -rf /tmp/labflow-demo

RUN install -m 755 docker/runpod/start.sh /usr/local/bin/labflow-pod-start \
    && ln -s /usr/local/bin/labflow-pod-start /start.sh \
    && install -m 755 docker/runpod/liteloc.sh /usr/local/bin/labflow-liteloc \
    && install -m 644 docker/runpod/sshd_config /etc/ssh/sshd_config_labflow \
    && printf 'PATH=%s\nPYTHONPATH=/opt/smlm-labflow\nMPLBACKEND=Agg\nLABFLOW_LEGACY_PYTHON=%s\n' "$PATH" "$LABFLOW_LEGACY_PYTHON" > /etc/environment

WORKDIR /workspace
EXPOSE 22
# RunPod may inject a shell command. Keep CMD independently executable instead
# of requiring arguments to pass through a custom init/entrypoint chain.
ENTRYPOINT []
CMD ["/start.sh"]
