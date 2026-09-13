"""Install pinned LiteLoc source and its isolated Linux GPU environment.

Reuse LABFLOW_LEGACY_PYTHON when supplied by the RunPod image; otherwise create
an environment using the checked-in conda explicit recipe. Never alter upstream
code or replace an existing checkout at a different revision.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/Li-Lab-SUSTech/LiteLoc.git"
SOURCE_REV = "aa4d262b89c7a872debaf908d82e1aaa26fe6915"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    config_path = ROOT / "adapters/backend_paths.yml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    config = config or {}
    source = Path(os.environ.get("LABFLOW_LITELOC_ROOT") or
                  (config.get("liteloc") or {}).get("root") or ROOT / "backends/LiteLoc").resolve()
    envroot = Path(os.environ.get("LABFLOW_ENV_ROOT", str(ROOT / "envs"))).resolve()
    python = Path(os.environ.get("LABFLOW_LEGACY_PYTHON", str(envroot / "liteloc/bin/python")))
    probe = ("import sys;sys.path.insert(0,sys.argv[1]); "
             "import torch,spline,hdfdict; from network.loc_model import LocModel; "
             "from network.multi_process import CompetitiveSmlmDataAnalyzer_multi_producer")
    def check():
        if not (source / ".git").exists() or not python.is_file():
            return False
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        return revision == SOURCE_REV and subprocess.run(
            [str(python), "-c", probe, str(source)], capture_output=True).returncode == 0
    if args.check:
        return 0 if check() else 1
    if sys.platform != "linux":
        raise RuntimeError("This pinned LiteLoc environment recipe currently supports Linux. Use a separately validated backend environment on other platforms.")
    if not (source / ".git").exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", SOURCE_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", SOURCE_REV], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REV:
        raise RuntimeError(f"Existing LiteLoc checkout is {revision}, expected {SOURCE_REV}. It has been left unchanged.")
    if not python.is_file():
        subprocess.run(["conda", "create", "-y", "-p", str(python.parent.parent), "--file",
                        str(ROOT / "env_yamls/liteloc_env_working_explicit.txt")], check=True)
    if not check():
        isolated = "import sys,pathlib;assert sys.prefix!=sys.base_prefix or (pathlib.Path(sys.prefix)/'conda-meta').is_dir(), 'LiteLoc requires an isolated environment'"
        subprocess.run([str(python), "-c", isolated], check=True)
        # Retain the known scientific/spline-compatible requirements while using
        # the Blackwell-compatible torch wheel family instead of CUDA 12.1.
        lines = (ROOT / "requirements/remote.txt").read_text().splitlines()
        lines = [line for line in lines if not line.startswith(("torch==", "torchvision==", "torchaudio==", "--find-links"))]
        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "scientific.txt"
            requirements.write_text("\n".join(lines) + "\n")
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements/blackwell.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "hdfdict==0.3.1"], check=True)
        # Same compatibility patch used by the image: avoid obsolete pkg_resources
        # dependency resolution in hdfdict's package initialization.
        patch = "from pathlib import Path;import site; p=Path(site.getsitepackages()[0])/'hdfdict/__init__.py'; p.write_text('from importlib.metadata import version\\nfrom .hdfdict import load,dump\\n__version__=version(\"hdfdict\")\\n')"
        subprocess.run([str(python), "-c", patch], check=True)
    if not check():
        raise RuntimeError("LiteLoc source/environment import check failed. Existing dependencies were left in place; inspect with the selected GPU interpreter.")
    config.setdefault("liteloc", {})["root"] = str(source)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    record = {"source": SOURCE_URL, "revision": SOURCE_REV, "python": str(python),
              "scope": "source and imports checked; GPU execution and model inference require separate checks"}
    (source.parent / "liteloc-install.json").write_text(json.dumps(record, indent=2))
    print(f"LiteLoc source and imports ready at {source}; interpreter: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
