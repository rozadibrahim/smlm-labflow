"""
labflow.conformance

End-to-end smoke test of every registered method over a tiny synthetic fixture.
It answers the lab's real question -- "does my install actually work, and which
tools can I run right now?" -- and doubles as the CI regression gate.

For each method the registry knows:
  - pick the synthetic fixture its stage consumes (localizations / clusters /
    tracks / image);
  - if the method's env / container / binary is not installed -> SKIP (clear,
    not a failure);
  - otherwise run it through `run_method` (which validates the output against the
    file contract and writes provenance) and record PASS / FAIL + elapsed.

A method that raises NotImplementedError is reported SKIP (a registered-but-unbuilt
scaffold), not FAIL. Stages that need real data (localize / calibrate / train /
report) have no synthetic fixture and are reported SKIP(no fixture).

Run:
    labflow conformance                 # every installed method
    labflow conformance --stage cluster
    python -m labflow.conformance
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .install import is_installed
from .registry import load_registry
from .registry import methods as reg_methods
from .registry import stages as reg_stages
from .registry import validate_registry
from .runner import run_method

# Which synthetic fixture each stage consumes. Stages absent here need real input
# (raw movies, a trained model) and are reported SKIP(no fixture).
STAGE_FIXTURE = {
    "drift": "drift_input",
    "render": "localizations",
    "track": "localizations",
    "cluster": "localizations",
    "spatial_stats": "localizations",
    "metrics": "localizations",
    "qc_audit": "localizations",
    "counting": "clusters",
    "analyze": "tracks",
    "phenotype": "clusters",
    "segment": "image",
    "test": "localizations",
}


def _make_localizations(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Two blobs across ~60 frames, with quality columns + a few outliers."""
    rng = np.random.default_rng(seed)
    half = n // 2
    xy = np.r_[rng.normal((100, 100), 8, (half, 2)),
               rng.normal((400, 400), 8, (n - half, 2))]
    frame = np.tile(np.arange(n // 4), 4)[:n]
    return pd.DataFrame({
        "frame": frame, "x": xy[:, 0], "y": xy[:, 1],
        "z": rng.normal(0.0, 5.0, n),
        "photons": np.r_[rng.normal(600, 40, n - 5), [50.0, 40, 9000, 8000, 30]],
        "sigma": rng.normal(1.2, 0.1, n),
    })


def _make_drift_input(seed: int = 0) -> pd.DataFrame:
    """Persistent emitters across the FOV, blinking over many frames, shifted by a
    known drift trajectory -- the structure rcc / aim / dme actually cross-correlate.
    Real-data scale on purpose: a couple of tight blobs is too sparse for drift
    estimation, and AIM needs the frame count to exceed its track_interval (it aligns
    >=2 time segments), so this uses ~1200 frames so the methods' default params work.
    """
    rng = np.random.default_rng(seed)
    emitters = rng.uniform(500, 5500, (200, 2))           # fixed super-res structure
    # bright fiducials present in EVERY frame, at cell centers so the (modest) drift
    # keeps each one inside one detection cell -> the fiducial backend can find them.
    fiducials = np.array([[1550.0, 1550.0], [4550.0, 2550.0], [3050.0, 5050.0]])
    n_frames = 1200                                        # >> track_interval (real scale)
    f = np.arange(n_frames)
    drift = np.c_[15.0 * np.sin(2 * np.pi * f / n_frames), 40.0 * f / n_frames]  # smooth, known
    rows = []
    for fi in range(n_frames):
        idx = rng.choice(len(emitters), 5, replace=False)  # a few emitters blink each frame
        for x, y in emitters[idx] + drift[fi] + rng.normal(0, 5, (5, 2)):
            rows.append({"frame": fi, "x": float(x), "y": float(y), "photons": 600.0})
        for x, y in fiducials + drift[fi] + rng.normal(0, 1.0, fiducials.shape):
            rows.append({"frame": fi, "x": float(x), "y": float(y), "photons": 5000.0})
    return pd.DataFrame(rows)


def _make_clusters(seed: int = 0) -> pd.DataFrame:
    """Two blinking clusters (0 every 2 frames, 1 every 3) -> qPAINT influx differs."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid, step in ((0, 2), (1, 3)):
        for f in range(0, 60, step):
            rows.append({"frame": f, "x": float(rng.normal(100 * (cid + 1), 3)),
                         "y": float(rng.normal(100 * (cid + 1), 3)), "cluster_id": cid})
    return pd.DataFrame(rows)


def _make_tracks() -> pd.DataFrame:
    rows = []
    for tid in range(3):
        for f in range(20):
            rows.append({"track_id": tid, "frame": f,
                         "x": 100 + tid * 50 + f * 2.0, "y": 200 - tid * 30 + f * 1.0,
                         "photons": 600.0})
    return pd.DataFrame(rows)


def _write_fixtures(d: Path) -> Dict[str, Path]:
    fx: Dict[str, Path] = {}
    fx["localizations"] = d / "localizations.csv"
    _make_localizations().to_csv(fx["localizations"], index=False)
    fx["drift_input"] = d / "drift_input.csv"
    _make_drift_input().to_csv(fx["drift_input"], index=False)
    fx["clusters"] = d / "clusters.csv"
    _make_clusters().to_csv(fx["clusters"], index=False)
    fx["tracks"] = d / "tracks.csv"
    _make_tracks().to_csv(fx["tracks"], index=False)
    try:                                  # image fixture only if tifffile is present
        import tifffile
        yy, xx = np.mgrid[0:128, 0:128]
        img = np.random.default_rng(0).poisson(50, (128, 128)).astype("uint16")
        for cx, cy in [(40, 40), (90, 80), (60, 100)]:
            img += (3000 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 9.0 ** 2)))).astype("uint16")
        fx["image"] = d / "image.tif"
        tifffile.imwrite(fx["image"], img)
    except Exception:
        pass                              # segment tools then report SKIP(no fixture)
    return fx


@dataclass
class Result:
    name: str
    stage: str
    runtime: str
    status: str                # PASS | FAIL | SKIP
    detail: str = ""
    elapsed: Optional[float] = None


def run_conformance(stage: Optional[str] = None, reg=None) -> List[Result]:
    """Run the smoke test and return one Result per method (grouped by stage)."""
    reg = reg if reg is not None else load_registry()
    # Structurally-incomplete entries (a planned scaffold with no command/entry/env)
    # are not runnable -- skip them with the lint reason rather than crashing.
    unrunnable = {name: msg for _lvl, name, msg in validate_registry(reg)}
    results: List[Result] = []
    with TemporaryDirectory(prefix="labflow_conf_") as tmp:
        d = Path(tmp)
        fx = _write_fixtures(d)
        for st in reg_stages(reg):
            if stage and st != stage:
                continue
            for name, spec in reg_methods(st, reg).items():
                s = dict(spec)
                s["name"] = name
                rt = str(s.get("runtime", "python"))
                if name in unrunnable:
                    results.append(Result(name, st, rt, "SKIP", unrunnable[name]))
                    continue
                kind = STAGE_FIXTURE.get(st)
                if not kind or kind not in fx:
                    results.append(Result(name, st, rt, "SKIP",
                                          "no synthetic fixture (needs real data)"))
                    continue
                if not is_installed(s):
                    results.append(Result(name, st, rt, "SKIP", "not installed (needs its env)"))
                    continue
                inp = fx[kind]
                out = d / f"out_{name}{inp.suffix}"
                t0 = time.perf_counter()
                try:
                    run_method(name, input_path=str(inp), output_path=str(out), reg=reg)
                    dt = round(time.perf_counter() - t0, 2)
                    if Path(out).exists():
                        results.append(Result(name, st, rt, "PASS", "", dt))
                    else:
                        results.append(Result(name, st, rt, "FAIL", "no output produced", dt))
                except NotImplementedError as exc:
                    results.append(Result(name, st, rt, "SKIP",
                                          "not implemented: " + _first_line(exc)))
                except Exception as exc:
                    results.append(Result(name, st, rt, "FAIL", _first_line(exc),
                                          round(time.perf_counter() - t0, 2)))
    return results


def _first_line(exc: Exception) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0][:120] if text else exc.__class__.__name__)


def print_report(results: List[Result]) -> int:
    """Print a grouped report; return the number of FAILs (the exit code)."""
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    print("\nlabflow conformance - end-to-end smoke test on synthetic fixtures\n")
    cur = None
    for r in results:
        if r.stage != cur:
            cur = r.stage
            print(f"[{r.stage}]")
        counts[r.status] = counts.get(r.status, 0) + 1
        t = f"  ({r.elapsed}s)" if r.elapsed is not None else ""
        print(f"  {r.status:4s} {r.name:16s} {r.runtime:8s} {r.detail}{t}")
    print(f"\n  {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    if counts["FAIL"]:
        print("  (a FAIL is an installed, ready tool that did not produce a valid output)")
    print()
    return counts["FAIL"]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="labflow conformance",
                                 description="Smoke-test installed methods end-to-end.")
    ap.add_argument("--stage", default=None, help="Only test methods of this stage.")
    args = ap.parse_args(argv)
    return print_report(run_conformance(args.stage))


if __name__ == "__main__":
    raise SystemExit(main())
