#!/usr/bin/env bash
# Run manually after SSH login; never install dependencies during Pod startup.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gpu_python=${LABFLOW_LEGACY_PYTHON:-/opt/conda/envs/smlm-labflow/bin/python}
env_dir=$("$gpu_python" -c 'import sys; print(sys.prefix)')
if [[ "$env_dir" != /opt/conda/envs/smlm-labflow ]]; then
    echo 'This repair targets the RunPod image environment /opt/conda/envs/smlm-labflow.' >&2
    exit 1
fi
run_dir=$(mktemp -d /workspace/outputs/blackwell-upgrade-XXXXXXXX)
exec > >(tee "$run_dir/upgrade.log") 2>&1
echo "Upgrade record: $run_dir"
"$gpu_python" -m pip freeze > "$run_dir/before.freeze.txt"
"$gpu_python" -m pip check > "$run_dir/before.pip-check.txt" 2>&1 || true
# Keep a full backup at its original prefix layout for a rename-based rollback.
# It is container-local: both the live environment and this backup are lost on
# container replacement. Keep the recipe and result records on /workspace.
backup_dir=$(mktemp -d /opt/labflow-cu121-backup-XXXXXXXX)
cp -a --reflink=auto "$env_dir" "$backup_dir/environment"
printf '%s\n' "$backup_dir/environment" > "$run_dir/backup-path.txt"
echo "Environment backup: $backup_dir/environment"
"$gpu_python" -m pip install --only-binary=:all: --no-cache-dir \
    -r "$repo_dir/requirements/blackwell.txt"
"$gpu_python" -m pip freeze > "$run_dir/after.freeze.txt"
"$gpu_python" -m pip check > "$run_dir/after.pip-check.txt" 2>&1 || true
if ! diff -u "$run_dir/before.pip-check.txt" "$run_dir/after.pip-check.txt"; then
    echo 'Dependency check changed; inspect the recorded differences before using the environment.' >&2
    exit 1
fi
"$gpu_python" - <<'PY'
import spline, numpy, scipy, hdfdict, h5py, omegaconf, thop
assert numpy.__version__ == '1.24.4', numpy.__version__
print('Existing spline and scientific dependencies still import; NumPy remains 1.24.4.')
PY
"$gpu_python" "$repo_dir/scripts/check_gpu.py" | tee "$run_dir/gpu-check.json"
echo 'Blackwell runtime checks passed. Real-data model validation is a separate step.'
