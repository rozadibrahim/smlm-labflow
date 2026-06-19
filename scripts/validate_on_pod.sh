#!/usr/bin/env bash
# =============================================================================
# validate_on_pod.sh - turnkey GPU validation for SMLM LabFlow on a Linux pod.
#
# Run on a fresh Linux GPU pod (RunPod, etc.):
#     git clone https://github.com/rozadibrahim/smlm-labflow && cd smlm-labflow
#     bash scripts/validate_on_pod.sh
#
# Proves, on real hardware:
#   - the in-core backends run (labflow doctor + conformance)
#   - the GPU is visible
#   - per-tool isolation works (Apptainer-pull of the GHCR images, or Docker if the
#     pod has a daemon)
#   - real segmentation: Cellpose (PyTorch) + StarDist (TensorFlow)
#   - the CONFLICT TEST: two mutually-incompatible DL stacks in one session
#   - the Linux core lock is generated (commit requirements/core.linux.lock.txt)
#
# Honest scope: this validates the *plumbing + the segmenters*. DL tools with binding
# points (DECODE training, MAGIK, MIRO, DeepTRACE, ...) need their models wired first.
#
# Prereqs: the GHCR images must be built (push a v* tag) AND public, and Python >=3.11
# must be available (labflow core needs it; pods often ship 3.10 -> install python3.11).
#
# Reproducibility knobs (env vars):
#   APPTAINER_VERSION=1.3.6   pin the exact Apptainer release (else apt/PPA latest)
#   PY=python3.11             interpreter for the labflow core env
# The exact engine version that ran is printed in step 3 regardless.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python3.11}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
PASS=0; FAIL=0; SKIP=0
ok(){ echo "  [PASS] $*"; PASS=$((PASS+1)); }
no(){ echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
sk(){ echo "  [skip] $*"; SKIP=$((SKIP+1)); }

echo "== 0. prerequisites =="
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "  $PY not found. Install it, e.g.:  $SUDO apt-get install -y python3.11 python3.11-venv"
  exit 1
fi
nvidia-smi -L 2>/dev/null && ok "GPU visible" || sk "no GPU detected (CPU-only run)"

echo "== 1. labflow core env =="
if "$PY" bootstrap.py --python "$PY" >/tmp/boot.log 2>&1; then ok "core env created"; else
  no "bootstrap failed"; tail -5 /tmp/boot.log; exit 1; fi
# shellcheck disable=SC1091
source envs/labflow/bin/activate

echo "== 2. in-core verification (must pass) =="
labflow doctor || true
if labflow conformance >/tmp/conf.log 2>&1; then
  ok "conformance: $(grep -oE '[0-9]+ passed, [0-9]+ failed' /tmp/conf.log | tail -1)"
else no "conformance"; tail -8 /tmp/conf.log; fi

echo "== 3. container engine (a standard pod has no docker daemon) =="
# Reproducibility: set APPTAINER_VERSION (e.g. 1.3.6) to install that EXACT release from
# the official .deb; otherwise the distro's apt/PPA build (latest) is used. Either way the
# exact engine version that ran is recorded below.
APPTAINER_VERSION="${APPTAINER_VERSION:-}"
if docker info >/dev/null 2>&1; then ENGINE=docker
elif command -v apptainer >/dev/null 2>&1; then ENGINE=apptainer
else
  echo "  installing apptainer (daemonless, works on pods)${APPTAINER_VERSION:+ v$APPTAINER_VERSION}..."
  if [ -n "$APPTAINER_VERSION" ]; then
    deb="apptainer_${APPTAINER_VERSION}_amd64.deb"
    url="https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/${deb}"
    { curl -fsSL -o "/tmp/$deb" "$url" && $SUDO apt-get install -y -qq "/tmp/$deb"; } >/dev/null 2>&1 || true
  fi
  command -v apptainer >/dev/null 2>&1 || \
  { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq apptainer; } >/dev/null 2>&1 || \
  { $SUDO add-apt-repository -y ppa:apptainer/ppa && $SUDO apt-get update -qq && \
    $SUDO apt-get install -y -qq apptainer; } >/dev/null 2>&1 || true
  command -v apptainer >/dev/null 2>&1 && ENGINE=apptainer || ENGINE=none
fi
export LABFLOW_CONTAINER_ENGINE="$ENGINE"
ver="n/a"
[ "$ENGINE" = "apptainer" ] && ver="$(apptainer --version 2>/dev/null || echo unknown)"
[ "$ENGINE" = "docker" ] && ver="$(docker --version 2>/dev/null || echo unknown)"
echo "  engine = $ENGINE ($ver)"
[ -n "$APPTAINER_VERSION" ] && [ "$ENGINE" = "apptainer" ] && \
  case "$ver" in *"$APPTAINER_VERSION"*) ok "apptainer pinned to $APPTAINER_VERSION";; *) sk "apptainer pin requested ($APPTAINER_VERSION) but got: $ver";; esac

echo "== 4. synthetic test image =="
python - <<'PY'
import numpy as np, tifffile
yy, xx = np.mgrid[0:256, 0:256]
img = np.random.default_rng(0).poisson(40, (256, 256)).astype("uint16")
for cx, cy in [(60, 60), (180, 90), (120, 190), (200, 210)]:
    img += (4000 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0 ** 2)))).astype("uint16")
tifffile.imwrite("/tmp/cells.tif", img); print("  wrote /tmp/cells.tif")
PY

echo "== 5. real segmentation (Cellpose [torch] + StarDist [TF]) =="
if [ "$ENGINE" = "none" ]; then
  sk "no container engine -> docker tools cannot run on this pod"
else
  for tool in cellpose stardist; do
    if labflow install "$tool" >/tmp/inst_$tool.log 2>&1; then
      if labflow run segment -b "$tool" -i /tmp/cells.tif -o /tmp/masks_$tool.tif >/tmp/run_$tool.log 2>&1; then
        n=$(python -c "import tifffile;print(int(tifffile.imread('/tmp/masks_$tool.tif').max()))" 2>/dev/null || echo "?")
        ok "$tool segmented -> $n objects"
      else no "$tool run (see /tmp/run_$tool.log)"; tail -4 /tmp/run_$tool.log; fi
    else no "$tool install/pull (image public? see /tmp/inst_$tool.log)"; tail -4 /tmp/inst_$tool.log; fi
  done
fi

echo "== 6. CONFLICT TEST: two incompatible DL stacks in one session =="
# Cellpose (PyTorch) and StarDist (TensorFlow) each ran in their own isolated container
# over the file contract. If both produced masks, two frameworks that cannot share a
# Python process coexisted in one pipeline -- the core isolation guarantee, on real GPU.
if [ -s /tmp/masks_cellpose.tif ] && [ -s /tmp/masks_stardist.tif ]; then
  ok "conflict-free: torch (cellpose) + TF (stardist) both ran in one pipeline"
else
  sk "conflict test needs both cellpose + stardist to have segmented above"
fi

echo "== 7. Linux core lock (commit requirements/core.linux.lock.txt) =="
if "$PY" bootstrap.py --lock >/dev/null 2>&1 && [ -f requirements/core.linux.lock.txt ]; then
  ok "Linux lock generated"
else sk "lock not generated"; fi

echo ""
echo "============================================================"
echo "  RESULT: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"
[ "$FAIL" -eq 0 ]
