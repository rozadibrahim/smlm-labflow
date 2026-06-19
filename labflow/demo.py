"""
labflow.demo

`labflow demo` -- run the in-core pipeline end-to-end on SYNTHETIC data, with zero
microscope data and no heavy installs. It proves an install actually works and shows the
shape of a run (drift -> cluster -> counting -> spatial_stats -> metrics -> render, plus
track -> analyze). Uses only the light in-core backends, so it needs the `.[light]` extra
(scikit-learn, trackpy, scipy, tifffile).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def run_demo(outdir: str = "demo_out") -> str:
    from .conformance import _make_drift_input, _make_localizations, _make_tracks
    from .runner import run_method

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # synthetic inputs (no microscope data needed)
    locs = out / "localizations.csv"; _make_localizations().to_csv(locs, index=False)
    drift_in = out / "drift_input.csv"; _make_drift_input().to_csv(drift_in, index=False)

    results = []

    def step(stage, method, inp, out_name, **params):
        try:
            run_method(method, input_path=str(inp), output_path=str(out / out_name), params=params)
            results.append(("ok", f"{stage} -b {method}", out_name))
        except Exception as exc:  # noqa: BLE001 - demo reports, doesn't abort
            results.append(("SKIP", f"{stage} -b {method}", str(exc).splitlines()[0][:70]))

    step("drift", "rcc", drift_in, "drift_corrected.csv")
    step("cluster", "dbscan", locs, "clusters.csv", eps=120, min_samples=8)
    if (out / "clusters.csv").exists():
        step("counting", "qpaint", out / "clusters.csv", "counts.csv")
    step("spatial_stats", "ripley", locs, "ripley.csv")
    step("metrics", "frc", drift_in, "metrics_frc.csv", render_nm=20)
    step("metrics", "nena", drift_in, "metrics_nena.csv")
    step("render", "render", locs, "sr_image.tif", render_nm=20)
    step("track", "trackpy", locs, "tracks_linked.csv", search_range=400)
    if (out / "tracks_linked.csv").exists():
        step("analyze", "msd", out / "tracks_linked.csv", "track_analysis.csv")

    n_ok = sum(1 for s, _, _ in results if s == "ok")
    print(f"\nlabflow demo -> {out.resolve()}\n")
    for status, what, info in results:
        print(f"  {status:4s}  {what:22s} {info}")
    print(f"\n{n_ok}/{len(results)} stages produced output. "
          f"Open it:  labflow review {out} --with napari")
    return str(out)
