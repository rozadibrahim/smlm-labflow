"""
labflow.runner

Execute a registered method over the canonical file contract. One entry point,
`run_method`, dispatches by the method's `runtime`:

    python   - import a callable `module:function` and run it in-process
    external - run a command directly with the given paths (e.g. localization,
               which takes a movie folder / run dir, not a CSV)
    local    - subprocess on PATH, CSV-in / CSV-out via a temp workspace
    venv     - like local, with a per-method virtualenv prepended to PATH
    conda    - like local, wrapped in `conda run -n <env>`
    docker   - run in the method's image, CSV mounted at /work

This is the single mechanism that docks drift correctors, trackers, analysers,
and external tools in any language, keeping their environments isolated.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .registry import REPO_ROOT, resolve

_PJSON = "\x00PARAMS_JSON\x00"


def _import_callable(entry: str):
    module, _, func = entry.partition(":")
    if not func:
        raise ValueError(f"python entry must be 'module:function', got {entry!r}")
    return getattr(importlib.import_module(module), func)


def _fmt(template: List[str], ctx: Dict[str, Any], params_json: str) -> List[str]:
    out: List[str] = []
    for part in template:
        s = str(part).replace("{params_json}", _PJSON).format(**ctx).replace(_PJSON, params_json)
        out.append(s)
    return out


def _check(proc: subprocess.CompletedProcess, name: str, args) -> None:
    if proc.returncode != 0:
        err = (proc.stderr or "")[-2000:] if proc.stderr else ""
        out = (proc.stdout or "")[-800:] if proc.stdout else ""
        raise RuntimeError(
            f"method {name} failed (rc={proc.returncode})\ncmd: {args}\n"
            f"stderr:\n{err}\nstdout:\n{out}"
        )


def _run_external(spec, merged, input_path, output_path) -> str:
    ctx = {"input": str(input_path), "output": str(output_path),
           "repo": str(REPO_ROOT), "p": SimpleNamespace(**merged)}
    args = _fmt(spec["command"], ctx, json.dumps(merged, separators=(",", ":")))
    proc = subprocess.run(args, cwd=str(REPO_ROOT))
    _check(proc, spec["name"], args)
    return str(output_path)


def _run_subprocess(spec, runtime, merged, input_path, output_path) -> str:
    in_name = spec.get("input", "in.csv")
    out_name = spec.get("output", "out.csv")
    pjson = json.dumps(merged, separators=(",", ":"))

    with tempfile.TemporaryDirectory(prefix=f"labflow_{spec['name']}_") as tmp:
        work = Path(tmp)
        shutil.copy(input_path, work / in_name)
        env: Optional[Dict[str, str]] = None
        cwd: Optional[str] = None

        if runtime == "docker":
            image = spec.get("image")
            if not image:
                raise ValueError(f"method {spec['name']}: runtime docker needs an image.")
            ctx = {"input": f"/work/{in_name}", "output": f"/work/{out_name}",
                   "repo": "/repo", "p": SimpleNamespace(**merged)}
            inner = _fmt(spec["command"], ctx, pjson)
            gpu = bool(spec.get("gpu"))
            from .install import _sif, engine
            if engine() == "apptainer":               # daemonless / HPC
                args = (["apptainer", "exec", "--containall"] + (["--nv"] if gpu else [])
                        + ["--bind", f"{work}:/work", "--pwd", "/work",
                           str(_sif(spec["name"]))] + inner)
            else:
                args = (["docker", "run", "--rm"] + (["--gpus", "all"] if gpu else [])
                        + ["-v", f"{work}:/work", "-w", "/work", image] + inner)
        else:
            ctx = {"input": str(work / in_name), "output": str(work / out_name),
                   "repo": str(REPO_ROOT), "p": SimpleNamespace(**merged)}
            args = _fmt(spec["command"], ctx, pjson)
            cwd = str(work)
            if runtime == "venv":
                env_dir = Path(spec["env"])
                bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
                env = os.environ.copy()
                env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
            elif runtime == "conda":
                args = ["conda", "run", "-n", spec["env"]] + args
            elif runtime != "local":
                raise ValueError(f"method {spec['name']}: unknown runtime {runtime!r}.")

        proc = subprocess.run(args, cwd=cwd, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        _check(proc, spec["name"], args)

        produced = work / out_name
        if not produced.exists():
            raise RuntimeError(
                f"method {spec['name']} exited 0 but did not write {out_name}.\n"
                f"stdout:\n{(proc.stdout or '')[-800:]}"
            )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(produced, out)
    return str(output_path)


def run_method(
    name: str,
    *,
    input_path: str | Path,
    output_path: str | Path,
    params: Optional[Dict[str, Any]] = None,
    reg: Optional[Dict[str, Any]] = None,
) -> str:
    spec = resolve(name, reg)

    if str(spec.get("status", "ready")).lower() != "ready":
        from .install import is_installed
        if not is_installed(spec):
            raise RuntimeError(
                f"method '{name}' (stage {spec.get('stage')}) needs its environment "
                f"first. Install it (automatic, isolated):\n"
                f"    labflow install {name}\n  {spec.get('description', '')}"
            )

    runtime = str(spec.get("runtime", "python")).lower()
    merged = {**(spec.get("params") or {}), **(spec.get("bind") or {}), **(params or {})}

    if runtime == "python":
        fn = _import_callable(spec["entry"])
        fn(input_csv=str(input_path), output_csv=str(output_path), params=dict(merged))
    elif runtime == "external":
        _run_external(spec, merged, input_path, output_path)
    else:
        _run_subprocess(spec, runtime, merged, input_path, output_path)

    _finalize(spec, runtime, merged, input_path, output_path)
    return str(output_path)


def _finalize(spec, runtime, merged, input_path, output_path) -> None:
    """Enforce the file contract and record provenance for the produced output.

    The two halves of "clean, conflict-free": validate makes the boundary the next
    isolated tool reads trustworthy; provenance makes the run reproducible. Either
    can be skipped with LABFLOW_NO_VALIDATE / LABFLOW_NO_PROVENANCE.
    """
    if not os.environ.get("LABFLOW_NO_VALIDATE"):
        from .contract import validate
        validate(output_path, spec.get("stage"), role="output")
    if not os.environ.get("LABFLOW_NO_PROVENANCE"):
        from . import provenance
        from .install import engine
        eng = engine() if runtime == "docker" else None
        provenance.write(output_path, spec=spec, params=merged,
                         input_path=input_path, runtime=runtime, engine=eng)
