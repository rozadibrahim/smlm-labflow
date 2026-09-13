"""Install optional backends in isolated environments from registry recipes.

A native install succeeds only after dependency checks and target-interpreter
imports pass. A receipt and resolved package inventory are kept in the environment.
This is installation evidence; conformance and reference checks establish whether
an adapter actually works. Docker/Apptainer paths remain available for registered
container methods. Unimplemented adapters fail before downloading dependencies.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
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
    root = Path(os.environ.get("LABFLOW_ENV_ROOT", str(REPO_ROOT / "envs"))).resolve()
    # Accept both historical spellings: 'miro' and 'envs/miro'.
    if p.parts and p.parts[0] == "envs":
        p = Path(*p.parts[1:])
    return root / p


def _recipe_hash(spec):
    inst = spec.get("install", {})
    payload = {"install": inst}
    if inst.get("requirements"):
        path = REPO_ROOT / inst["requirements"]
        payload["requirements_content"] = path.read_text() if path.is_file() else None
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _tool_env(spec):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    models = str(Path(os.environ.get("LABFLOW_MODEL_ROOT", str(REPO_ROOT / "models"))).resolve())
    env.update({str(k): str(v).replace("{models}", models)
                for k, v in spec.get("environment", {}).items()})
    return env


def _probe_venv(spec, *, explain=False):
    probes = (spec.get("install") or {}).get("probe", [])
    probes = [probes] if isinstance(probes, str) else probes
    py = _venv_python(_envdir(spec.get("env", spec["name"])))
    if not py.exists():
        return False
    # Import in the target interpreter, never infer success from a directory.
    code = "import importlib,json,sys; [importlib.import_module(x) for x in json.loads(sys.argv[1])]"
    try:
        proc = subprocess.run([str(py), "-c", code, json.dumps(probes)],
                              env=_tool_env(spec), capture_output=True, text=True, timeout=90)
        if explain and proc.returncode:
            print((proc.stderr or proc.stdout)[-3000:])
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


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
    if spec.get("implementation") == "stub":
        return False
    runtime = str(spec.get("runtime", "python")).lower()
    script = (spec.get("install") or {}).get("script")
    if script:
        return _run([sys.executable, str(REPO_ROOT / script), "--check"])

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
        receipt = _envdir(spec.get("env", spec["name"])) / ".labflow-install.json"
        try:
            recorded = json.loads(receipt.read_text())
            return recorded.get("recipe_hash") == _recipe_hash(spec) and _probe_venv(spec)
        except (OSError, ValueError):
            return False

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
        if not src.exists():
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

    if inst.get("script"):
        return [[sys.executable, str(REPO_ROOT / inst["script"])]]

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
        cmds.append([str(_venv_python(envdir)), "-m", "pip", "check"])

    elif runtime == "conda":
        env = spec.get("env", name)
        cf = inst.get("conda_file")
        if cf:                                  # build the env from the spec file
            if not (REPO_ROOT / cf).is_file():
                raise ValueError(f"{name}: missing environment recipe: {cf}")
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
    if spec.get("implementation") == "stub":
        raise RuntimeError(f"{name}: adapter is not implemented; installing dependencies cannot make it runnable. "
                           + str(spec.get("description", "")))
    runtime = str(spec.get("runtime", "python")).lower()
    inst = spec.get("install") or {}

    rc = inst.get("requires_cmd")
    if rc and shutil.which(rc) is None:
        raise RuntimeError(f"{name}: needs the external '{rc}' binary on PATH. "
                           f"{spec.get('description', '')}")

    if runtime in IN_CORE and not inst.get("extra") and not inst.get("script"):
        if not is_installed(spec):
            raise RuntimeError(f"{name}: required backend is absent and no automatic install recipe is available.")
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
    if runtime == "venv" and not dry_run:
        (_envdir(spec.get("env", name)) / ".labflow-install.json").unlink(missing_ok=True)
    for cmd in cmds:
        print("  $ " + " ".join(cmd))
        if not dry_run:
            if subprocess.run(cmd, cwd=str(REPO_ROOT), env=_tool_env(spec)).returncode != 0:
                raise RuntimeError(f"install step failed for {name}: {' '.join(cmd)}")
    if not dry_run:
        if runtime == "venv":
            if not _probe_venv(spec, explain=True):
                raise RuntimeError(f"{name}: dependencies installed but backend import checks failed.")
            envdir = _envdir(spec.get("env", name))
            freeze = subprocess.check_output([str(_venv_python(envdir)), "-m", "pip", "freeze"], text=True)
            (envdir / ".labflow-freeze.txt").write_text(freeze)
            (envdir / ".labflow-install.json").write_text(json.dumps({
                "method": name, "recipe_hash": _recipe_hash(spec), "python": str(_venv_python(envdir))}, indent=2))
        print(f"{name}: installed. Run:  labflow run {spec.get('stage')} -b {name} ...")
