"""
labflow.registry

Single source of truth for every method in the pipeline. Methods are declared in
config/methods.yaml; this module loads, lists, and resolves them. Both the CLI
(labflow.cli) and the Snakemake workflow read from here, so adding a method is one
YAML entry that shows up in both places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "config" / "methods.yaml"

# Canonical stage order (each stage consumes the previous stage's CSV contract).
DEFAULT_STAGES = ["calibrate", "train", "localize", "drift", "render", "segment",
                  "track", "cluster", "spatial_stats", "counting", "phenotype",
                  "analyze", "metrics", "qc_audit", "report"]


def load_registry(path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(path) if path else DEFAULT_REGISTRY
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "methods" not in data:
        raise ValueError(f"{path}: missing 'methods' section.")
    return data


def stages(reg: Optional[Dict[str, Any]] = None) -> List[str]:
    reg = reg if reg is not None else load_registry()
    declared = list(reg.get("stages", DEFAULT_STAGES))
    # include any stage referenced by a method but not declared (e.g. 'test')
    extra = sorted({m.get("stage", "") for m in reg.get("methods", {}).values()}
                   - set(declared) - {""})
    return declared + extra


def methods(stage: Optional[str] = None,
            reg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    reg = reg if reg is not None else load_registry()
    items = reg.get("methods", {}) or {}
    if stage is None:
        return items
    return {k: v for k, v in items.items() if v.get("stage") == stage}


def resolve(name: str, reg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    reg = reg if reg is not None else load_registry()
    items = reg.get("methods", {}) or {}
    if name not in items:
        raise KeyError(f"Unknown method {name!r}. Known: {sorted(items)}")
    spec = dict(items[name])
    spec["name"] = name
    return spec


KNOWN_RUNTIMES = ("python", "external", "local", "venv", "conda", "docker")


def validate_registry(reg: Optional[Dict[str, Any]] = None) -> List[tuple]:
    """Lint every method entry so a malformed plug fails fast and clearly.

    Returns a list of (level, method, message). `error` = the method cannot run
    as written; `warn` = an intentionally incomplete scaffold (status != ready)
    still missing what it needs to run. An empty list means a clean registry.
    Used by `labflow doctor`; cheap enough to call in tests/CI.
    """
    reg = reg if reg is not None else load_registry()
    declared = set(stages(reg))
    issues: List[tuple] = []

    for name, spec in (reg.get("methods", {}) or {}).items():
        stage = spec.get("stage")
        runtime = str(spec.get("runtime", "python")).lower()
        # A not-yet-ready scaffold may legitimately still lack its command/env.
        lvl = "error" if str(spec.get("status", "ready")).lower() == "ready" else "warn"

        if not stage:
            issues.append(("error", name, "no 'stage' declared"))
        elif stage not in declared:
            issues.append(("error", name, f"stage {stage!r} is not in the registry's stages"))

        if runtime not in KNOWN_RUNTIMES:
            issues.append(("error", name, f"unknown runtime {runtime!r} (use one of {KNOWN_RUNTIMES})"))
            continue

        # Completeness gaps use `lvl` (error only if the method claims to be ready;
        # a planned scaffold is allowed to still be missing its command/env/image).
        # Structurally-invalid values (bad stage/runtime, malformed entry) are always
        # errors regardless of status.
        if runtime == "python":
            entry = spec.get("entry")
            if not entry:
                issues.append((lvl, name, "python runtime needs 'entry: module:function'"))
            elif ":" not in str(entry):
                issues.append(("error", name, f"entry {entry!r} must be 'module:function'"))
        elif runtime in ("external", "local"):
            if not spec.get("command"):
                issues.append((lvl, name, f"{runtime} runtime needs a 'command' arg-list"))
        elif runtime in ("venv", "conda"):
            if not spec.get("env"):
                issues.append((lvl, name, f"{runtime} runtime needs 'env'"))
            if not spec.get("command"):
                issues.append((lvl, name, f"{runtime} runtime needs a 'command' before it can run"))
        elif runtime == "docker":
            if not spec.get("image") and not (spec.get("install") or {}).get("context"):
                issues.append((lvl, name, "docker runtime needs 'image' or install.context"))
            if not spec.get("command"):
                issues.append((lvl, name, "docker runtime needs a 'command' before it can run"))

    return issues
