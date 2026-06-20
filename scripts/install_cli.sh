#!/usr/bin/env bash
# Install a global `labflow` command on your PATH that delegates to THIS repo's isolated
# labflow environment -- so you can run `labflow ...` from any directory, in bash, without
# activating the venv or cd-ing into the project.
#
#   bash scripts/install_cli.sh                         # -> ~/.local/bin/labflow
#   LABFLOW_BIN=/usr/local/bin sudo -E bash scripts/install_cli.sh   # system-wide
#
# Cleaner cross-platform alternative for other labs (isolated + on PATH automatically):
#   pipx install smlm-labflow        # once it's on PyPI
#   pipx install -e .                # from a clone
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Locate the installed `labflow` entry point inside the repo's env (Windows vs Unix).
LAUNCH=""
for cand in "$REPO/envs/labflow/Scripts/labflow.exe" "$REPO/envs/labflow/bin/labflow"; do
  if [ -x "$cand" ]; then LAUNCH="$cand"; break; fi
done
# Fall back to `python -m labflow` if the entry-point exe isn't present.
PYBIN=""
if [ -z "$LAUNCH" ]; then
  for cand in "$REPO/envs/labflow/Scripts/python.exe" "$REPO/envs/labflow/bin/python"; do
    if [ -x "$cand" ]; then PYBIN="$cand"; break; fi
  done
  if [ -z "$PYBIN" ]; then
    echo "labflow env not found under $REPO/envs/labflow. Create it first:  python bootstrap.py" >&2
    exit 1
  fi
fi

BINDIR="${LABFLOW_BIN:-$HOME/.local/bin}"
mkdir -p "$BINDIR"
TARGET="$BINDIR/labflow"
if [ -n "$LAUNCH" ]; then
  printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$LAUNCH" > "$TARGET"
else
  printf '#!/usr/bin/env bash\nexec "%s" -m labflow "$@"\n' "$PYBIN" > "$TARGET"
fi
chmod +x "$TARGET"
echo "installed: $TARGET"

case ":$PATH:" in
  *":$BINDIR:"*) echo "ready -> run from anywhere:  labflow help" ;;
  *) echo "Add $BINDIR to PATH (once):"
     echo "  echo 'export PATH=\"$BINDIR:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
esac
