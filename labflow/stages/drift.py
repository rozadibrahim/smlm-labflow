"""
labflow.stages.drift

In-process adapter for the drift stage. Reuses driftcorr unchanged: it loads the
localizations, runs the selected backend (rcc / aim_julia / dme / none), applies
the correction, and writes the canonical drift outputs. The `backend` param
selects the method, so all drift methods share this one adapter.

Contract: localizations CSV in -> drift_corrected_localizations.csv out
(with drift_trajectory.csv and drift_correction.json written alongside).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from ..io import read_localizations


def run(*, input_csv: str, output_csv: str, params: Dict[str, Any]) -> str:
    from driftcorr.core import apply_drift, write_outputs
    from driftcorr.registry import get_backend

    p = dict(params or {})
    backend = p.pop("backend", "rcc")
    pixel = p.pop("pixel_size_nm", None)
    units = p.pop("units", "nm")

    locs = read_localizations(input_csv)
    estimate = get_backend(backend)(locs, pixel_size_nm=pixel, units=units, params=p)
    corrected = apply_drift(locs, estimate)

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_outputs(
        out.parent, backend=backend, locs=locs, corrected=corrected,
        estimate=estimate, source_csv=str(input_csv),
        pixel_size_nm=pixel, units=units,
    )

    produced = out.parent / "drift_corrected_localizations.csv"
    if produced.resolve() != out.resolve():
        shutil.copy(produced, out)
    return str(out)
