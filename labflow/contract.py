"""
labflow.contract

Validate the canonical file contract at a stage boundary.

The isolation guarantee — tools never share a Python process; the only thing that
crosses a tool boundary is a file — is only *meaningful* if those files actually
conform. Two perfectly isolated tools still "conflict" if tool A emits `X,Y` and
tool B reads `x,y`: the failure surfaces as a confusing crash deep inside the next
tool. This module turns that into a clear, located error at the boundary.

We enforce the *consumed subset*: the columns the next stage actually reads (not
every canonical column), so optional metadata never causes a false failure while a
genuinely broken contract (missing coordinates / ids) is always caught. The full
canonical schema lives in `schema.py`; these mirror its REQUIRED_* lists.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

# Stage -> columns its CSV output MUST carry for the next stage to consume it.
# Stages absent here have no canonical-CSV contract (image / run-level / non-tabular:
# calibrate, train, segment, spatial_stats, counting, phenotype, qc_audit, report).
STAGE_REQUIRED = {
    "localize": ["frame", "x", "y"],
    "drift":    ["frame", "x", "y"],
    "track":    ["track_id", "frame", "x", "y"],
    "cluster":  ["frame", "x", "y", "cluster_id"],
    "analyze":  ["track_id"],
}

_CSV_SUFFIXES = (".csv", ".tsv", ".txt")


class ContractError(ValueError):
    """A stage output does not satisfy the file contract the next stage reads."""


def required_for(stage: Optional[str]) -> Optional[List[str]]:
    """The columns `stage` must emit, or None if the stage has no CSV contract."""
    return STAGE_REQUIRED.get(str(stage)) if stage else None


def _header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        row = next(csv.reader(fh), [])
    return [c.strip() for c in row]


def validate(path, stage, *, role: str = "output") -> None:
    """Raise ContractError if `path` (a CSV produced by `stage`) is non-conforming.

    Segmentation has an integer-label TIFF contract. Other stages without a
    declared CSV contract are left to their adapter-specific checks.
    """
    if stage == "segment":
        import numpy as np
        import tifffile
        p = Path(path)
        try:
            mask = tifffile.imread(p)
        except Exception as exc:
            raise ContractError(f"{role} of segment is not a readable TIFF: {p}") from exc
        if mask.ndim not in (2, 3) or not np.issubdtype(mask.dtype, np.integer) or mask.size == 0 or mask.min() < 0:
            raise ContractError(f"{role} of segment must be a nonempty 2D/3D array of nonnegative integer labels: {p}")
        return
    req = required_for(stage)
    if not req:
        return
    p = Path(path)
    if not p.exists():
        raise ContractError(f"{role} for stage '{stage}' is missing: {p}")
    if p.suffix.lower() not in _CSV_SUFFIXES:
        return
    header = _header(p)
    if not header:
        raise ContractError(f"{role} {p} is empty (stage '{stage}').")
    missing = [c for c in req if c not in header]
    if missing:
        raise ContractError(
            f"{role} of stage '{stage}' breaks the file contract: missing "
            f"column(s) {missing}.\n  got:      {header}\n  required: {req}\n"
            f"The next stage reads these columns — fix the method's output mapping."
        )
