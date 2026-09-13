#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/conda/envs/smlm-labflow/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
export CONDA_DEFAULT_ENV=smlm-labflow
cd /opt/smlm-labflow
exec /opt/conda/envs/smlm-labflow/bin/python run_pipeline.py "$@"
