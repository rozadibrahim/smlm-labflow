"""
labflow.install

`labflow install <tool>` — automatic, reproducible, isolated dependency
installation, designed so a biologist never resolves or builds a dependency.

Best path (heavy tools): a prebuilt, version-pinned image is *pulled* (not built)
from a registry — the dependency-error class disappears because nothing is solved
on the user's machine. Light pure-python tools install as a pip extra.

Container engine is selectable (so labs without a Docker daemon, or HPC, work):
    LABFLOW_CONTAINER_ENGINE=docker   (default)
    LABFLOW_CONTAINER_ENGINE=apptainer

A method's `install:` block (+ `runtime`) declares how:
    runtime: docker  install: {pull: true}            -> docker/apptainer pull {image}
    runtime: docker  install: {context: docker/<t>}   -> docker build (local, fallback)
    runtime: venv    install: {pip: [...], git: ...}  -> envs/<env> venv
    runtime: conda   install: {conda_file: ...}       -> conda env
    runtime: python  install: {extra, probe}          -> pip install ".[extra]"
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .registry import REPO_ROOT, resolve

IN_CORE = ("python", "external", "local")


def engine() -> str:
    return os.environ.get("LABFLOW_CONTAINER_ENGINE", "docker").lower()


def _in_isolated_env() -> bool:
    """True when running inside a venv or conda env (not the global interpreter)."""
    return sys.prefix != sys.base_prefix or bool(os.environ.get("CONDA_PREFIX"))


def _envdir(env: str) -> Path:
    p = Path(env)
    if p.is_absolute():
        return p
    return REPO_ROOT / p if ("/" in env or os.sep in env) else REPO_ROOT / "envs" / env


def _venv_python(envdir: Path) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    return envdir / sub / ("python.exe" if os.name == "nt" else "python")


def _sif(name: str) -> Path:
    return REPO_ROOT / "envs" / "sif" / f"{name}.sif"


def _run(cmd: List[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def is_installed(spec: Dict[str, Any]) -> bool:
    runtime = str(spec.get("runtime", "python")).lower()

    # A required external binary (e.g. julia, fiji) must be on PATH regardless of
    # runtime -- lets a tool that shells out report "needs its env" cleanly instead
    # of failing mid-run.
    requires_cmd = (spec.get("install") or {}).get("requires_cmd")
    if requires_cmd and shutil.which(requires_cmd) is None:
        return False

    if runtime == "docker":
        if engine() == "apptainer":
            return _sif(spec["name"]).exists()
        image = spec.get("image", "")
        return bool(image) and _run(["docker", "image", "inspect", image])

    if runtime == "venv":
        return _venv_python(_envdir(spec.get("env", ""))).exists()

    if runtime == "conda":
        env = spec.get("env", "")
        try:
            r = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        except FileNotFoundError:
            return False
        return r.returncode == 0 and any(
            line.split() and line.split()[0] == env for line in (r.stdout or "").splitlines())

    # python / external / local: the entry module must actually exist on disk
    # (catches registered-but-unimplemented stubs), then any probe dependency.
    entry = spec.get("entry")
    if entry:
        try:
            if importlib.util.find_spec(entry.split(":", 1)[0]) is None:
                return False
        except ModuleNotFoundError:
            return False
    probe = (spec.get("install") or {}).get("probe")
    return importlib.util.find_spec(probe) is not None if probe else True


def _source_cmds(inst: Dict[str, Any], src: Path, pip: List[str]) -> List[List[str]]:
    """Fetch + install a tool's OWN source (not just its deps) -- shared by the venv
    and conda paths. `src` is where the repo is cloned; `pip` is that env's
    pip-install prefix. This is what lets a spec file "contain the tool": git clone +
    requirements + pip [-e], all declared in the method's `install:` block.
    """
    cmds: List[List[str]] = []
    if inst.get("git"):
        cmds.append(["git", "clone", "--depth", "1", inst["git"], str(src)])
    if inst.get("requirements"):
        cmds.append(pip + ["-r", str(REPO_ROOT / inst["requirements"])])
    if inst.get("requirements_in_src"):
        cmds.append(pip + ["-r", str(src / inst["requirements_in_src"])])
    if inst.get("pip"):
        cmds.append(pip + list(inst["pip"]))
    if inst.get("editable"):
        cmds.append(pip + ["-e", str(src)])
    return cmds


def plan(spec: Dict[str, Any], *, build: bool = False) -> List[List[str]]:
    """Exact commands `install` will run (what --dry-run prints)."""
    runtime = str(spec.get("runtime", "python")).lower()
    inst = spec.get("install") or {}
    name = spec["name"]
    cmds: List[List[str]] = []

    if runtime == "docker":
        image = spec.get("image", "")
        if engine() == "apptainer":                     # daemonless / HPC
            cmds.append(["apptainer", "pull", str(_sif(name)), f"docker://{image}"])
        elif build or not inst.get("pull"):             # build locally
            ctx = inst.get("context")
            if not ctx:
                raise ValueError(f"{name}: no install.context to build (and pull not set).")
            cmds.append(["docker", "build", "-t", image, str(REPO_ROOT / ctx)])
        else:                                           # pull prebuilt (the one-liner)
            cmds.append(["docker", "pull", image])

    elif runtime == "venv":
        envdir = _envdir(spec.get("env", name))
        pip = [str(_venv_python(envdir)), "-m", "pip", "install"]
        cmds.append([sys.executable, "-m", "venv", str(envdir)])
        cmds += _source_cmds(inst, envdir.parent / f"{envdir.name}_src", pip)

    elif runtime == "conda":
        env = spec.get("env", name)
        cf = inst.get("conda_file")
        if cf:                                  # build the env from the spec file
            cmds.append(["conda", "env", "create", "-n", env, "-f", str(REPO_ROOT / cf)])
        else:                                   # or a bare env at a chosen python
            cmds.append(["conda", "create", "-y", "-n", env,
                         f"python={inst.get('python', '3.9')}"])
        # then (optionally) fetch + install the tool's own source into that env
        pip = ["conda", "run", "-n", env, "python", "-m", "pip", "install"]
        cmds += _source_cmds(inst, REPO_ROOT / "envs" / f"{name}_src", pip)

    else:
        extra = inst.get("extra")
        if extra:
            cmds.append([sys.executable, "-m", "pip", "install", "-e", f".[{extra}]"])
    return cmds


def install_tool(name: str, *, dry_run: bool = False, force: bool = False,
                 build: bool = False, reg=None) -> None:
    spec = resolve(name, reg)
    runtime = str(spec.get("runtime", "python")).lower()
    inst = spec.get("install") or {}

    rc = inst.get("requires_cmd")
    if rc and shutil.which(rc) is None:
        print(f"{name}: needs the external '{rc}' binary on PATH (not auto-installable). "
              f"Obtain it and add it to PATH. {spec.get('description', '')}")
        return

    if runtime in IN_CORE and not inst.get("extra"):
        extra_note = f" (uses the '{rc}' binary)" if rc else ""
        print(f"{name}: runtime '{runtime}', no extra deps - already in the core env{extra_note}.")
        return
    if not force and is_installed(spec):
        print(f"{name}: already installed.")
        return

    cmds = plan(spec, build=build)
    if not cmds:
        print(f"{name}: nothing to install (no install spec).")
        return

    # Guard: a `python` extra installs into the *current* interpreter. Refuse to do
    # that to a global Python — that's how the core env gets polluted and tools
    # start to conflict. venv/conda/docker installs are self-isolated, so exempt.
    if (runtime in IN_CORE and not dry_run and not _in_isolated_env()
            and not os.environ.get("LABFLOW_ALLOW_GLOBAL")):
        raise RuntimeError(
            f"{name}: refusing to install into the global interpreter\n"
            f"  ({sys.prefix}).\n"
            f"Create the isolated core env first, then run labflow from it:\n"
            f"    python -m venv envs/labflow\n"
            f"    envs/labflow/Scripts/pip install -e \".[light]\"   # Windows\n"
            f"    #   envs/labflow/bin/pip install -e \".[light]\"   # macOS/Linux\n"
            f"Override (not recommended): set LABFLOW_ALLOW_GLOBAL=1."
        )

    eng = f", {engine()}" if runtime == "docker" else ""
    print(f"installing '{name}' ({runtime}{eng}){' [dry-run]' if dry_run else ''}:")
    for cmd in cmds:
        print("  $ " + " ".join(cmd))
        if not dry_run:
            cwd = str(REPO_ROOT) if cmd[0] in (sys.executable, "docker", "apptainer") else None
            if subprocess.run(cmd, cwd=cwd).returncode != 0:
                raise RuntimeError(f"install step failed for {name}: {' '.join(cmd)}")
    if not dry_run:
        print(f"{name}: installed. Run:  labflow run {spec.get('stage')} -b {name} ...")
