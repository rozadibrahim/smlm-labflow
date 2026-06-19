"""
driftcorr.registry

The single extension point. To add a drift-correction backend, write a function
with the signature::

    estimate(locs, *, pixel_size_nm, units, params) -> DriftEstimate

and add it to DRIFT_BACKENDS below. Everything downstream (Snakemake rule,
report, benchmark) is backend-agnostic and needs no changes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

from .aim_julia import estimate_drift_aim_julia
from .core import DriftEstimate
from .dme import estimate_drift_dme
from .fiducial import estimate_drift_fiducial
from .rcc import estimate_drift_rcc

DriftBackend = Callable[..., DriftEstimate]


def estimate_drift_none(
    locs: pd.DataFrame,
    *,
    pixel_size_nm: Optional[float] = None,
    units: str = "nm",
    params: Optional[Dict[str, Any]] = None,
) -> DriftEstimate:
    """Passthrough: no correction (preserves current LabFlow proxy behaviour)."""
    return DriftEstimate.zero(locs, units=units, method="none")


DRIFT_BACKENDS: Dict[str, DriftBackend] = {
    "none": estimate_drift_none,
    "rcc": estimate_drift_rcc,
    "fiducial": estimate_drift_fiducial,    # track bright persistent markers (classical)
    "aim_julia": estimate_drift_aim_julia,  # faithful Julia port of the authors' AIM
    "dme": estimate_drift_dme,
}


def get_backend(name: str) -> DriftBackend:
    key = str(name or "none").strip().lower()
    if key not in DRIFT_BACKENDS:
        raise KeyError(
            f"Unknown drift backend {name!r}. "
            f"Available: {sorted(DRIFT_BACKENDS)}."
        )
    return DRIFT_BACKENDS[key]
