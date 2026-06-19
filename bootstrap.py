#!/usr/bin/env python3
"""
bootstrap.py — materialize labflow's isolated core env on ANY machine.

The project carries its own setup: the same command bootstraps a laptop, a CI
runner, or a rented GPU pod. Pure standard library; Linux / macOS / Windows.

    python bootstrap.py                  # create envs/labflow, install the light core
    python bootstrap.py --extras light,gui
    python bootstrap.py --python python3.11   # build the env from a specific interpreter
    python bootstrap.py --from-lock      # reproduce exactly from this platform's lock
    python bootstrap.py --lock           # (re)generate this platform's lockfile, then exit
    python bootstrap.py --no-venv        # install into the current interpreter
                                         #   (use INSIDE an already-isolated container/pod)
    python bootstrap.py --dry-run        # print the commands, change nothing

After it finishes:  labflow doctor

Note: labflow core needs Python >=3.11 (snakemake 8). Many GPU pods ship 3.10, so
labflow gets its own env there — the heavy GPU tools stay in the pod's torch env /
their own containers. That separation is the whole point.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "envs" / "labflow"
MIN_PY = (3, 11)


def venv_python(env: Path) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    return env / sub / ("python.exe" if os.name == "nt" else "python")


def platform_tag() -> str:
    return {"Linux": "linux", "Darwin": "macos", "Windows": "win"}.get(
        platform.system(), platform.system().lower())


def lock_path() -> Path:
    return ROOT / "requirements" / f"core.{platform_tag()}.lock.txt"


def _interpreter_version(exe: str) -> tuple:
    try:
        out = subprocess.run(
            [exe, "-c", "import sys;print('%d %d' % sys.version_info[:2])"],
            capture_output=True, text=True, check=True).stdout.split()
        return (int(out[0]), int(out[1]))
    except Exception:
        return (0, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize the labflow core env on any machine.")
    ap.add_argument("--extras", default="light", help="extra group(s), comma-separated (default: light)")
    ap.add_argument("--python", default=sys.executable, help="interpreter to build the venv from")
    ap.add_argument("--no-venv", action="store_true", help="install into the current interpreter (containers/pods)")
    ap.add_argument("--from-lock", action="store_true", help="install pinned from this platform's lockfile first")
    ap.add_argument("--lock", action="store_true", help="(re)generate this platform's lockfile, then exit")
    ap.add_argument("-n", "--dry-run", action="store_true", help="print commands, change nothing")
    args = ap.parse_args()

    base_exe = sys.executable if args.no_venv else args.python
    ver = _interpreter_version(base_exe)
    if ver and ver < MIN_PY:
        print(f"warning: {base_exe} is Python {ver[0]}.{ver[1]}, but labflow core needs "
              f">={MIN_PY[0]}.{MIN_PY[1]} (snakemake 8).")
        print(f"  get a {MIN_PY[0]}.{MIN_PY[1]} interpreter and pass it, e.g.:")
        print(f"    uv venv --python {MIN_PY[0]}.{MIN_PY[1]} envs/labflow   # or conda/pyenv")
        print(f"    python bootstrap.py --python <that python>")

    cmds: list = []
    if args.no_venv:
        py = Path(sys.executable)
        print("note: --no-venv installs into the current interpreter; only do this "
              "inside an already-isolated container/pod.")
    else:
        py = venv_python(ENV)
        if not py.exists():
            cmds.append([base_exe, "-m", "venv", str(ENV)])

    pip = [str(py), "-m", "pip"]
    cmds.append(pip + ["install", "--upgrade", "pip", "wheel"])

    lp = lock_path()
    if args.from_lock:
        if lp.exists():
            cmds.append(pip + ["install", "-r", str(lp)])
        else:
            print(f"no lock for this platform yet ({lp.name}); run --lock on this OS first.")
    cmds.append(pip + ["install", "-e", f".[{args.extras}]"])

    where = "current interpreter" if args.no_venv else ENV.name
    suffix = f"  (then freeze -> {lp.name})" if args.lock else ""
    print(f"\nbootstrap labflow ({platform_tag()}, {where}){' [dry-run]' if args.dry_run else ''}:{suffix}")
    for c in cmds:
        print("  $ " + " ".join(c))
        if not args.dry_run:
            subprocess.run(c, cwd=str(ROOT), check=True)

    if args.lock:
        if args.dry_run:
            print(f"  $ {py} -m pip freeze  > requirements/{lp.name}")
            return 0
        out = subprocess.run([str(py), "-m", "pip", "freeze"], cwd=str(ROOT),
                             capture_output=True, text=True, check=True).stdout
        lines = [ln for ln in out.splitlines()
                 if ln and not ln.startswith("-e ") and "smlm-labflow" not in ln.lower()]
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {lp.relative_to(ROOT)}  ({len(lines)} pinned packages)")
        return 0

    if not args.dry_run:
        print("\nlabflow core ready.")
        if not args.no_venv:
            act = f"{ENV}\\Scripts\\activate" if os.name == "nt" else f"source {ENV}/bin/activate"
            print(f"  activate:  {act}")
        print("  verify:    labflow doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
