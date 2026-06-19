"""
driftcorr: pluggable drift-correction backends for SMLM LabFlow.

A drift backend is a single callable with a uniform contract::

    estimate(locs, *, pixel_size_nm, units, params) -> DriftEstimate

where ``locs`` is a DataFrame of canonical localizations (frame, x, y[, z,
lpx, lpy]) and ``DriftEstimate`` carries a per-frame drift trajectory.
Backends are registered in ``driftcorr.registry.DRIFT_BACKENDS`` and selected
by name, which is the single extension point: add a localization-microscopy
drift method by writing one function and registering it.

Shipped backends:
    none  - passthrough (no correction; current LabFlow proxy behaviour)
    rcc   - redundant cross-correlation (Wang et al. 2014), pure numpy
    dme   - drift at minimum entropy (Cnossen et al. 2021), state of the art
"""

from .core import DriftEstimate, apply_drift, load_localizations, write_outputs
from .registry import DRIFT_BACKENDS, get_backend

__all__ = [
    "DriftEstimate",
    "apply_drift",
    "load_localizations",
    "write_outputs",
    "DRIFT_BACKENDS",
    "get_backend",
]
