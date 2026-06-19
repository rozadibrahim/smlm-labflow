"""
labflow.provenance

A reproducibility manifest written beside every output (`<output>.labflow.json`).

Isolation makes a run *clean*; provenance makes it *auditable*. For each output we
record exactly what produced it — method, runtime, isolation level, the image
digest (or env), the parameters, and content hashes of the input and output. Two
labs can then prove they ran the same thing, or pinpoint precisely what differed.
This is the difference between "reproducible in principle" and "reproducible".
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_BUF = 1 << 20


def _sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_BUF), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except OSError:
        return None  # e.g. a movie *directory* input — not a single file


def _labflow_version() -> str:
    try:
        from importlib.metadata import version
        return version("smlm-labflow")
    except Exception:
        return "0+unknown"


def _image_digest(image: str) -> Optional[str]:
    """The immutable repo digest of a local image, if the engine can report it."""
    if not image:
        return None
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "--format",
             "{{index .RepoDigests 0}}", image],
            capture_output=True, text=True)
    except FileNotFoundError:
        return None
    out = (r.stdout or "").strip()
    return out or None


def _isolation_of(runtime: str, spec: Dict[str, Any]) -> str:
    if runtime in ("python", "external", "local"):
        return "core-extra" if (spec.get("install") or {}).get("extra") else "in-core"
    return {"venv": "venv", "conda": "conda", "docker": "container"}.get(runtime, runtime)


def manifest_path(output_path) -> Path:
    p = Path(output_path)
    return p.with_name(p.name + ".labflow.json")


def write(output_path, *, spec: Dict[str, Any], params: Dict[str, Any],
          input_path, runtime: str, engine: Optional[str] = None) -> Path:
    """Write the manifest next to `output_path`; return its path."""
    out = Path(output_path)
    rec: Dict[str, Any] = {
        "labflow_version": _labflow_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": spec.get("stage"),
        "method": spec.get("name"),
        "runtime": runtime,
        "isolation": _isolation_of(runtime, spec),
        "params": params,
        "input": {"path": str(input_path), "sha256": _sha256(Path(input_path))},
        "output": {"path": str(out), "sha256": _sha256(out)},
        "host": {"platform": platform.platform(),
                 "python": sys.version.split()[0]},
    }
    if runtime == "docker":
        img = spec.get("image", "")
        rec["image"] = {"ref": img, "digest": _image_digest(img),
                        "engine": engine, "gpu": bool(spec.get("gpu"))}
    elif runtime in ("venv", "conda"):
        rec["env"] = spec.get("env")

    mp = manifest_path(out)
    mp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return mp
