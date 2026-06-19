"""
driftcorr.run

Uniform CLI for every drift backend. This is what the Snakemake rule shells out
to, so all backends share one invocation contract::

    python -m driftcorr.run --backend rcc \
        --locs results/batches/<id>/canonical_localizations.csv \
        --out  results/batches/<id>/drift \
        --pixel-size 127 --units nm \
        --param n_time_bins=25 --param render_nm=20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from .core import apply_drift, load_localizations, write_outputs
from .registry import DRIFT_BACKENDS, get_backend


def _parse_params(pairs) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            params[key] = json.loads(value)        # numbers, bools, lists
        except json.JSONDecodeError:
            params[key] = value                     # plain string
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a pluggable drift-correction backend to localizations."
    )
    parser.add_argument("--backend", default="rcc", choices=sorted(DRIFT_BACKENDS))
    parser.add_argument("--locs", required=True, help="Localization CSV (canonical or raw).")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--pixel-size", type=float, default=None, help="Pixel size (nm).")
    parser.add_argument("--units", default="nm", choices=["nm", "pixel"])
    parser.add_argument("--param", action="append", default=None,
                        help="Backend parameter key=value (JSON value). Repeatable.")
    args = parser.parse_args()

    params = _parse_params(args.param)
    estimate_fn = get_backend(args.backend)

    locs = load_localizations(args.locs)
    started = time.perf_counter()
    estimate = estimate_fn(
        locs, pixel_size_nm=args.pixel_size, units=args.units, params=params
    )
    elapsed = time.perf_counter() - started
    corrected = apply_drift(locs, estimate)

    outputs = write_outputs(
        args.out,
        backend=args.backend,
        locs=locs,
        corrected=corrected,
        estimate=estimate,
        source_csv=str(Path(args.locs).resolve()),
        pixel_size_nm=args.pixel_size,
        units=args.units,
        elapsed_sec=elapsed,
    )

    metrics = estimate.residual_metrics()
    print(f"drift backend : {args.backend} ({estimate.method})")
    print(f"localizations : {len(locs):,}  frames: {estimate.frames.size:,}")
    print(f"max radial    : {metrics.get('max_radial_drift')!r} {estimate.units}")
    print(f"elapsed       : {elapsed:.2f} s")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
