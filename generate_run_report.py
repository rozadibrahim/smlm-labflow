#!/usr/bin/env python3
"""
generate_run_report.py

Generate a human-readable report for one SMLM pipeline run.

Works with two on-disk layouts:

1. Pipeline run directories (batch layout)
       run_summary.json
       batch_manifest.csv
       runtime_benchmark.csv
       batches/<id>/canonical_localizations.csv ...

2. Benchmark / comparison directories (benchmark layout)
       benchmark_summary.json
       comparison_ready_summary.json
       machine_specs.json
       runtime_benchmark.csv / resource_benchmark.csv
       localization_qc_benchmark.csv
       resolution_benchmark.csv  +  frc_curve_batch_*.csv
       drift_benchmark.csv
       export_validation.csv

The report pulls in *every* layer it can find, generates plots from the data
(runtime, localizations, FRC curve, drift), and writes a fully populated
Markdown report plus a self-contained HTML report so everything about a run
can be reviewed in one place.

Usage:
    python generate_run_report.py --run outputs/benchmark_comparison/thunderstorm_bench

Outputs:
    run_report.md
    run_report.html
    report_assets/
        runtime_by_stage.png
        localizations_per_movie.png
        frc_curve.png
        drift_radial.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib  # Force headless backend before pyplot to avoid Windows DLL
matplotlib.use("Agg")  # delay-load crashes (0xC06D007F) inside batch inference.
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------------------------------- #
# Basic IO helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def image_to_data_uri(path: Optional[Path]) -> str:
    """Base64 data URI for png/svg/jpg; empty string when unavailable."""
    if path is None or not Path(path).exists():
        return ""

    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }.get(suffix, "image/png")

    try:
        data = Path(path).read_bytes()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


# Backwards-compatible alias (older callers expect a png data URI).
def image_to_base64(path: Optional[Path]) -> str:
    return image_to_data_uri(path)


def rel_posix(path: Path, start: Path) -> str:
    """Relative path with forward slashes, for portable Markdown links."""
    try:
        return os.path.relpath(str(path), str(start)).replace(os.sep, "/")
    except Exception:
        return str(path)


# --------------------------------------------------------------------------- #
# Number / value formatting
# --------------------------------------------------------------------------- #
def fnum(value: Any, decimals: int = 3) -> Optional[str]:
    """Format a number with thousands separators; None when not numeric/NaN."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(f):
        return None

    if abs(f - round(f)) < 1e-9 and abs(f) < 1e15:
        return f"{int(round(f)):,}"

    return f"{f:,.{decimals}f}"


def col_median(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(series.median()) if not series.empty else None


# --------------------------------------------------------------------------- #
# Batch summary (pipeline / batch layout)
# --------------------------------------------------------------------------- #
def find_batch_dirs(run_dir: Path) -> List[Path]:
    batches_dir = run_dir / "batches"

    if not batches_dir.exists():
        return []

    return sorted(
        [p for p in batches_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def count_canonical_localizations(batch_dir: Path) -> Optional[int]:
    csv_path = batch_dir / "canonical_localizations.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        return int(len(df))
    except Exception:
        return None


def summarize_batches(run_dir: Path) -> pd.DataFrame:
    manifest_path = run_dir / "batch_manifest.csv"
    manifest = read_csv_safe(manifest_path)

    rows: List[Dict[str, Any]] = []

    if not manifest.empty:
        for _, row in manifest.iterrows():
            batch_dir_raw = row.get("run_dir", "")
            batch_dir = Path(str(batch_dir_raw)) if batch_dir_raw else None

            if batch_dir is not None and not batch_dir.is_absolute():
                batch_dir = (run_dir / batch_dir).resolve()

            if batch_dir is None or not batch_dir.exists():
                batch_dir = None

            n_locs = count_canonical_localizations(batch_dir) if batch_dir else None

            rows.append(
                {
                    "batch_index": row.get("batch_index", ""),
                    "run_id": row.get("run_id", ""),
                    "input_name": row.get("input_name", ""),
                    "qc_status": row.get("qc_status", ""),
                    "backend_status": row.get("backend_status", ""),
                    "canonical_status": row.get("canonical_status", ""),
                    "shape": row.get("shape", ""),
                    "axes": row.get("axes", ""),
                    "dtype": row.get("dtype", ""),
                    "n_frames_guess": row.get("n_frames_guess", ""),
                    "n_localizations": n_locs,
                    "run_dir": str(batch_dir) if batch_dir else "",
                }
            )

    else:
        for i, batch_dir in enumerate(find_batch_dirs(run_dir), start=1):
            qc = read_json(batch_dir / "input_qc.json")
            n_locs = count_canonical_localizations(batch_dir)

            rows.append(
                {
                    "batch_index": i,
                    "run_id": batch_dir.name,
                    "input_name": qc.get("input_name", ""),
                    "qc_status": qc.get("qc_status", ""),
                    "backend_status": "",
                    "canonical_status": (
                        "passed"
                        if (batch_dir / "canonical_localizations.csv").exists()
                        else ""
                    ),
                    "shape": qc.get("shape", ""),
                    "axes": qc.get("axes", ""),
                    "dtype": qc.get("dtype", ""),
                    "n_frames_guess": qc.get("n_frames_guess", ""),
                    "n_localizations": n_locs,
                    "run_dir": str(batch_dir),
                }
            )

    return pd.DataFrame(rows)


def select_existing_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    existing = [col for col in columns if col in df.columns]
    return df[existing] if existing else df


def single_row_kv(
    df: pd.DataFrame,
    drop: Tuple[str, ...] = (),
    drop_prefix: Tuple[str, ...] = ("fig_",),
    max_len: int = 200,
) -> pd.DataFrame:
    """Transpose a one-row DataFrame into a tidy Metric / Value table.

    Empty, NaN and blacklisted columns are dropped; long strings truncated.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    row = df.iloc[0]
    items: List[Dict[str, str]] = []

    for col in df.columns:
        if col in drop or any(col.startswith(p) for p in drop_prefix):
            continue

        value = row[col]

        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue

        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            continue

        if len(text) > max_len:
            text = text[: max_len - 1] + "…"

        items.append({"Metric": col, "Value": text})

    return pd.DataFrame(items)


# --------------------------------------------------------------------------- #
# Runtime normalisation
# --------------------------------------------------------------------------- #
def normalize_runtime_units(runtime_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure both elapsed_sec and elapsed_min exist regardless of source schema."""
    if runtime_df.empty:
        return runtime_df

    runtime_df = runtime_df.copy()
    elapsed_sec = pd.Series(pd.NA, index=runtime_df.index, dtype="float64")

    if "start_time" in runtime_df.columns and "end_time" in runtime_df.columns:
        try:
            start_time = pd.to_datetime(
                runtime_df["start_time"], errors="coerce", utc=True
            )
            end_time = pd.to_datetime(
                runtime_df["end_time"], errors="coerce", utc=True
            )
            elapsed_sec = (end_time - start_time).dt.total_seconds()
        except Exception:
            pass

    if "elapsed_sec" in runtime_df.columns:
        existing_sec = pd.to_numeric(runtime_df["elapsed_sec"], errors="coerce")
        elapsed_sec = elapsed_sec.fillna(existing_sec)
    if "elapsed_min" in runtime_df.columns:
        existing_min_to_sec = (
            pd.to_numeric(runtime_df["elapsed_min"], errors="coerce") * 60.0
        )
        elapsed_sec = elapsed_sec.fillna(existing_min_to_sec)

    runtime_df["elapsed_sec"] = elapsed_sec
    runtime_df["elapsed_min"] = elapsed_sec / 60.0
    return runtime_df


# Backwards-compatible alias.
def normalize_runtime_minutes(runtime_df: pd.DataFrame) -> pd.DataFrame:
    return normalize_runtime_units(runtime_df)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_runtime_plot(runtime_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if runtime_df.empty:
        return None

    runtime_df = normalize_runtime_units(runtime_df)

    if "stage" not in runtime_df.columns or "elapsed_sec" not in runtime_df.columns:
        return None

    grouped_sec = (
        runtime_df.groupby("stage", dropna=False)["elapsed_sec"]
        .sum()
        .sort_values(ascending=False)
    )

    if grouped_sec.empty or float(grouped_sec.sum()) <= 0.0:
        return None

    use_seconds = bool((grouped_sec.max() < 60.0) or (grouped_sec.median() < 60.0))
    if use_seconds:
        grouped = grouped_sec
        ylabel = "Total runtime (s)"
    else:
        grouped = grouped_sec / 60.0
        ylabel = "Total runtime (min)"

    plt.figure(figsize=(8, 4.5))
    grouped.plot(kind="bar", color="#4C78A8")
    plt.ylabel(ylabel)
    plt.xlabel("Pipeline stage")
    plt.title("Runtime by pipeline stage")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def make_localization_plot(batch_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if batch_df.empty or "n_localizations" not in batch_df.columns:
        return None

    df = batch_df.copy()
    df["n_localizations"] = pd.to_numeric(df["n_localizations"], errors="coerce")
    df = df.dropna(subset=["n_localizations"])

    if df.empty:
        return None

    labels = df["input_name"].fillna(df["run_id"]).astype(str)
    values = df["n_localizations"].astype(int)

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("Number of localizations")
    plt.xlabel("Input movie")
    plt.title("Canonical localizations per movie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def make_frc_plot(frc_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if frc_df.empty:
        return None

    needed = {"frequency_cycles_per_nm", "frc"}
    if not needed.issubset(frc_df.columns):
        return None

    df = frc_df.copy()
    df["frequency_cycles_per_nm"] = pd.to_numeric(
        df["frequency_cycles_per_nm"], errors="coerce"
    )
    df["frc"] = pd.to_numeric(df["frc"], errors="coerce")
    df = df.dropna(subset=["frequency_cycles_per_nm", "frc"])

    if df.empty:
        return None

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        df["frequency_cycles_per_nm"],
        df["frc"],
        color="#4C78A8",
        linewidth=1.6,
        label="FRC",
    )

    if "threshold" in df.columns:
        thr = pd.to_numeric(df["threshold"], errors="coerce").dropna()
        if not thr.empty:
            plt.axhline(
                float(thr.iloc[0]),
                color="#E45756",
                linestyle="--",
                linewidth=1.2,
                label=f"Threshold ({thr.iloc[0]:.3f})",
            )

    plt.ylabel("FRC")
    plt.xlabel("Spatial frequency (cycles / nm)")
    plt.title("Fourier ring correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def make_drift_plot(drift_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if drift_df.empty or "frame_mid" not in drift_df.columns:
        return None

    df = drift_df.copy()
    df["frame_mid"] = pd.to_numeric(df["frame_mid"], errors="coerce")
    bins = df.dropna(subset=["frame_mid"]).sort_values("frame_mid")

    if bins.empty or "radial_drift" not in bins.columns:
        return None

    bins["radial_drift"] = pd.to_numeric(bins["radial_drift"], errors="coerce")
    bins = bins.dropna(subset=["radial_drift"])

    if bins.empty:
        return None

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        bins["frame_mid"],
        bins["radial_drift"],
        color="#4C78A8",
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Radial drift",
    )

    for axis, color in (("dx", "#72B7B2"), ("dy", "#F58518")):
        if axis in bins.columns:
            vals = pd.to_numeric(bins[axis], errors="coerce")
            if vals.notna().any():
                plt.plot(
                    bins["frame_mid"],
                    vals,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    label=axis,
                )

    plt.ylabel("Drift (nm, centroid proxy)")
    plt.xlabel("Frame (bin midpoint)")
    plt.title("Drift proxy vs frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No data available._"

    shown = df.head(max_rows).copy()

    try:
        return shown.to_markdown(index=False)
    except Exception:
        return shown.to_string(index=False)


def make_html_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"

    return df.head(max_rows).to_html(index=False, escape=True)


# --------------------------------------------------------------------------- #
# QC preview cards (batch layout)
# --------------------------------------------------------------------------- #
def collect_preview_cards(batch_df: pd.DataFrame, max_cards: int = 12) -> str:
    if batch_df.empty or "run_dir" not in batch_df.columns:
        return ""

    cards = []

    for _, row in batch_df.head(max_cards).iterrows():
        batch_dir = Path(str(row.get("run_dir", "")))

        preview_b64 = image_to_data_uri(batch_dir / "input_preview.png")
        hist_b64 = image_to_data_uri(batch_dir / "input_histogram.png")

        if not preview_b64 and not hist_b64:
            continue

        title = str(row.get("input_name", row.get("run_id", "movie")))
        qc_status = str(row.get("qc_status", ""))
        canonical_status = str(row.get("canonical_status", ""))
        n_locs = row.get("n_localizations", "")

        preview_img = (
            f'<img src="{preview_b64}" alt="preview" />'
            if preview_b64
            else "<p><em>No preview image.</em></p>"
        )
        hist_img = (
            f'<img src="{hist_b64}" alt="histogram" />'
            if hist_b64
            else "<p><em>No histogram image.</em></p>"
        )

        cards.append(
            f"""
            <div class="card">
                <h3>{title}</h3>
                <p>
                    <strong>QC:</strong> {qc_status}<br>
                    <strong>Canonical:</strong> {canonical_status}<br>
                    <strong>Localizations:</strong> {n_locs}
                </p>
                <div class="image-row">
                    <div>{preview_img}</div>
                    <div>{hist_img}</div>
                </div>
            </div>
            """
        )

    return "\n".join(cards)


# --------------------------------------------------------------------------- #
# Machine specs + KPIs + figures
# --------------------------------------------------------------------------- #
def machine_spec_rows(ms: Dict[str, Any]) -> pd.DataFrame:
    if not ms:
        return pd.DataFrame()

    def g(*path: str, default: Any = None) -> Any:
        cur: Any = ms
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    gpu_devices = g("nvidia_smi", "devices", default=[]) or []
    gpu_names = ", ".join(str(d.get("name", "")) for d in gpu_devices if d.get("name"))
    gpu_mem = ", ".join(
        str(d.get("total_memory_gb", "")) for d in gpu_devices if d.get("total_memory_gb")
    )

    rows = [
        ("Hostname", g("hostname")),
        ("OS", g("os", "platform") if isinstance(g("os"), dict) else g("os")),
        ("CPU", g("cpu", "model")),
        ("CPU logical cores", g("cpu", "logical_cores")),
        ("CPU physical cores", g("cpu", "physical_cores")),
        ("RAM total (GB)", g("memory", "ram_total_gb")),
        ("GPU(s)", gpu_names or None),
        ("GPU memory (GB)", gpu_mem or None),
        ("NVIDIA driver", g("nvidia_smi", "driver_version") or g("nvml", "driver_version")),
        ("CUDA (reported)", g("nvidia_smi", "cuda_version_reported")),
        ("Torch", g("torch_cuda", "torch_version") or g("packages", "torch")),
        ("NumPy", g("packages", "numpy")),
        ("pandas", g("packages", "pandas")),
        ("Python", g("python", "version")),
        ("Captured at", g("captured_at")),
    ]

    clean = [(k, v) for k, v in rows if v not in (None, "", "[]", "None")]
    return pd.DataFrame([{"Metric": k, "Value": str(v)} for k, v in clean])


def build_kpis(
    bench_summary: Dict[str, Any],
    comp_summary: Dict[str, Any],
    locqc_df: pd.DataFrame,
    runtime_captured: bool,
) -> List[Tuple[str, str]]:
    comp = comp_summary or {}
    bench = bench_summary or {}

    status = comp.get("benchmark_status") or bench.get("status") or "—"
    density = col_median(locqc_df, "density_per_um2")
    total_timed = comp.get("total_timed_sec")
    rss = comp.get("max_rss_mb")
    frc_res = comp.get("median_frc_resolution_nm")

    timed_label = (
        fnum(total_timed) + " s"
        if (runtime_captured and fnum(total_timed) and total_timed)
        else "not captured"
    )

    kpis: List[Tuple[str, str]] = [
        ("Benchmark status", str(status)),
        ("Total localizations", fnum(comp.get("total_localizations")) or "—"),
        ("Localization batches", fnum(comp.get("n_localization_batches")) or "—"),
        ("Density (um^-2)", fnum(density) or "—"),
        (
            "Sampling-limited res (nm)",
            fnum(comp.get("median_sampling_limited_resolution_nm")) or "—",
        ),
        ("FRC resolution (nm)", fnum(frc_res) or "FRC not crossed"),
        ("Max radial drift (nm)", fnum(comp.get("median_max_radial_drift")) or "—"),
        ("Export validation", str(comp.get("export_validation_status") or "—")),
        ("Timed runtime", timed_label),
        ("Peak RSS (MB)", fnum(rss) or "not captured"),
        ("GPU", str(comp.get("gpu_names") or "—")),
    ]
    return kpis


def collect_run_figures(run_dir: Path) -> List[Tuple[Path, str]]:
    """Figures produced by the run itself (run_dir/figures), if any."""
    sub = run_dir / "figures"
    if not sub.is_dir():
        return []

    figs: List[Tuple[Path, str]] = []
    for p in sorted(sub.glob("*.png")) + sorted(sub.glob("*.svg")):
        figs.append((p, p.stem.replace("_", " ")))
    return figs


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def generate_markdown_report(
    run_dir: Path,
    summary: Dict[str, Any],
    batch_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    runtime_plot: Optional[Path],
    loc_plot: Optional[Path],
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    extras = extras or {}
    created_at = datetime.now().isoformat(timespec="seconds")

    comp_summary: Dict[str, Any] = extras.get("comp_summary", {})
    bench_summary: Dict[str, Any] = extras.get("bench_summary", {})
    machine_specs: Dict[str, Any] = extras.get("machine_specs", {})
    locqc_df: pd.DataFrame = extras.get("locqc_df", pd.DataFrame())
    resolution_df: pd.DataFrame = extras.get("resolution_df", pd.DataFrame())
    drift_df: pd.DataFrame = extras.get("drift_df", pd.DataFrame())
    export_df: pd.DataFrame = extras.get("export_df", pd.DataFrame())
    resource_df: pd.DataFrame = extras.get("resource_df", pd.DataFrame())
    frc_plot: Optional[Path] = extras.get("frc_plot")
    drift_plot: Optional[Path] = extras.get("drift_plot")
    kpis: List[Tuple[str, str]] = extras.get("kpis", [])
    figures: List[Tuple[Path, str]] = extras.get("figures", [])
    runtime_captured: bool = extras.get("runtime_captured", False)

    n_movies = len(batch_df)
    qc_passed = (
        int((batch_df["qc_status"] == "passed").sum())
        if not batch_df.empty and "qc_status" in batch_df.columns
        else 0
    )
    canonical_passed = (
        int((batch_df["canonical_status"] == "passed").sum())
        if not batch_df.empty and "canonical_status" in batch_df.columns
        else 0
    )

    runtime_df = normalize_runtime_units(runtime_df)

    lines: List[str] = [
        "# SMLM Pipeline Run Report",
        "",
        f"**Generated:** {created_at}",
        "",
        "## Run overview",
        "",
        f"- Run folder: `{run_dir}`",
        f"- Input: `{summary.get('input', comp_summary.get('benchmark_dir', ''))}`",
        f"- Profile: `{summary.get('profile_path', '')}`",
        f"- Backend: `{summary.get('backend_name', '')}`",
        f"- Hostname: `{comp_summary.get('hostname', machine_specs.get('hostname', ''))}`",
        f"- OS: `{comp_summary.get('os', '')}`",
    ]
    if n_movies:
        lines += [
            f"- Movies processed: **{n_movies}**",
            f"- QC passed: **{qc_passed}/{n_movies}**",
            f"- Canonical conversion passed: **{canonical_passed}/{n_movies}**",
        ]
    lines.append("")

    # KPIs
    if kpis:
        lines += ["## Key metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {label} | {value} |" for label, value in kpis]
        lines.append("")

    # System / machine specs
    spec_rows = machine_spec_rows(machine_specs)
    if not spec_rows.empty:
        lines += ["## System", "", dataframe_to_markdown(spec_rows), ""]

    # Runtime
    runtime_cols = [
        "stage", "batch_index", "elapsed_sec", "elapsed_min", "status",
        "rss_mb", "process_cpu_percent", "gpu_peak_memory_allocated_mb",
    ]
    lines += ["## Runtime", ""]
    if runtime_captured and not runtime_df.empty:
        lines += [
            dataframe_to_markdown(select_existing_columns(runtime_df, runtime_cols)),
            "",
        ]
        if runtime_plot is not None:
            lines += [
                f"![Runtime by stage]({rel_posix(runtime_plot, run_dir)})",
                "",
            ]
    else:
        lines += [
            "_No per-stage timing was captured for this run "
            "(`runtime_benchmark.csv` is empty: psutil/torch were unavailable)._",
            "",
        ]

    # Resource usage
    lines += ["## Resource usage", ""]
    if not resource_df.empty:
        lines += [dataframe_to_markdown(resource_df, max_rows=20), ""]
    else:
        lines += ["_No resource samples captured for this run._", ""]

    # Localization QC
    loc_kv = single_row_kv(locqc_df, drop=("benchmark_layer",))
    lines += ["## Localization QC", ""]
    lines += [dataframe_to_markdown(loc_kv), ""]
    if loc_plot is not None:
        lines += [f"![Localizations per movie]({rel_posix(loc_plot, run_dir)})", ""]

    # Resolution + FRC
    lines += ["## Resolution", ""]
    res_cols = ["metric", "value", "unit", "status", "notes"]
    lines += [
        dataframe_to_markdown(select_existing_columns(resolution_df, res_cols)),
        "",
    ]
    if frc_plot is not None:
        lines += [
            "### FRC curve",
            "",
            f"![FRC curve]({rel_posix(frc_plot, run_dir)})",
            "",
        ]

    # Drift
    drift_summary_cols = [
        "status", "method", "max_abs_dx", "max_abs_dy", "max_radial_drift",
        "median_radial_drift", "p95_radial_drift",
        "linear_radial_drift_slope_per_frame", "message",
    ]
    drift_summary = select_existing_columns(drift_df.head(1), drift_summary_cols)
    lines += ["## Drift proxy", ""]
    lines += [dataframe_to_markdown(single_row_kv(drift_summary)), ""]
    if drift_plot is not None:
        lines += [f"![Drift vs frame]({rel_posix(drift_plot, run_dir)})", ""]

    # Export validation
    export_cols = [
        "export_name", "exists", "rows", "columns_ok", "status", "message",
    ]
    lines += ["## Export validation", ""]
    lines += [
        dataframe_to_markdown(select_existing_columns(export_df, export_cols)),
        "",
    ]

    # Batch summary (batch layout only)
    if not batch_df.empty:
        batch_cols = [
            "batch_index", "input_name", "qc_status", "backend_status",
            "canonical_status", "shape", "axes", "dtype", "n_localizations",
        ]
        lines += [
            "## Batch summary",
            "",
            dataframe_to_markdown(select_existing_columns(batch_df, batch_cols)),
            "",
        ]

    # Figure gallery
    if figures:
        lines += ["## Figures", ""]
        for path, caption in figures:
            lines += [f"**{caption}**", "", f"![{caption}]({rel_posix(path, run_dir)})", ""]

    # Files produced
    lines += [
        "## Files produced",
        "",
        "- `run_report.md` / `run_report.html`",
        "- `report_assets/` (generated plots)",
        "- `benchmark_summary.json` / `comparison_ready_summary.json`",
        "- `machine_specs.json`",
        "- `runtime_benchmark.csv` / `resource_benchmark.csv`",
        "- `localization_qc_benchmark.csv` / `resolution_benchmark.csv` / `drift_benchmark.csv`",
        "- `export_validation.csv` / `frc_curve_batch_*.csv`",
        "",
    ]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
HTML_CSS = """
    :root { color-scheme: light; }
    body {
        font-family: -apple-system, Segoe UI, Arial, sans-serif;
        margin: 0; padding: 32px 40px 64px;
        line-height: 1.5; color: #1f2933; background: #f4f6f8;
    }
    h1 { margin: 0 0 4px; font-size: 1.9rem; }
    h2 { margin: 36px 0 12px; padding-top: 8px; font-size: 1.35rem;
         border-bottom: 2px solid #e1e5ea; }
    h3 { font-size: 1.05rem; }
    .subtle { color: #647280; font-size: 0.92rem; margin: 2px 0; }
    code { background: #eef1f4; padding: 2px 6px; border-radius: 4px;
           font-size: 0.85em; }
    .toc { background: #fff; border: 1px solid #e1e5ea; border-radius: 10px;
           padding: 12px 18px; margin: 20px 0; }
    .toc a { color: #2563a8; text-decoration: none; margin-right: 16px;
             white-space: nowrap; font-size: 0.9rem; }
    .toc a:hover { text-decoration: underline; }
    .summary-grid {
        display: grid; gap: 12px; margin: 18px 0 8px;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    }
    .metric { background: #fff; border: 1px solid #e1e5ea; border-left: 4px solid #4C78A8;
              border-radius: 10px; padding: 12px 14px; }
    .metric .label { font-size: 0.78rem; color: #647280; text-transform: uppercase;
                     letter-spacing: 0.03em; }
    .metric .value { font-size: 1.35rem; font-weight: 700; margin-top: 6px;
                     word-break: break-word; }
    table { border-collapse: collapse; width: 100%; background: #fff;
            margin: 12px 0 8px; font-size: 0.88rem; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    th, td { border: 1px solid #e1e5ea; padding: 7px 9px; text-align: left;
             vertical-align: top; }
    th { background: #eef1f4; position: sticky; top: 0; }
    .plot, .gallery img { max-width: 100%; border: 1px solid #e1e5ea;
            border-radius: 8px; background: #fff; padding: 6px; }
    .note { background: #fff7e6; border: 1px solid #ffe1a8; border-radius: 8px;
            padding: 10px 14px; color: #7a5a00; }
    .gallery { display: grid; gap: 18px;
               grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .gallery figure { margin: 0; background: #fff; border: 1px solid #e1e5ea;
                      border-radius: 10px; padding: 10px; }
    .gallery figcaption { font-size: 0.85rem; color: #475569; margin-top: 6px;
                          font-weight: 600; }
    .card { background: #fff; border: 1px solid #e1e5ea; border-radius: 12px;
            padding: 16px; margin: 16px 0; }
    .image-row { display: grid; gap: 12px;
                 grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    .image-row img { max-width: 100%; border: 1px solid #e1e5ea; border-radius: 8px; }
"""


def _html_section(section_id: str, title: str, body: str) -> str:
    return f'<h2 id="{section_id}">{title}</h2>\n{body}\n'


def generate_html_report(
    run_dir: Path,
    summary: Dict[str, Any],
    batch_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    runtime_plot: Optional[Path],
    loc_plot: Optional[Path],
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    extras = extras or {}
    created_at = datetime.now().isoformat(timespec="seconds")

    comp_summary: Dict[str, Any] = extras.get("comp_summary", {})
    machine_specs: Dict[str, Any] = extras.get("machine_specs", {})
    locqc_df: pd.DataFrame = extras.get("locqc_df", pd.DataFrame())
    resolution_df: pd.DataFrame = extras.get("resolution_df", pd.DataFrame())
    drift_df: pd.DataFrame = extras.get("drift_df", pd.DataFrame())
    export_df: pd.DataFrame = extras.get("export_df", pd.DataFrame())
    resource_df: pd.DataFrame = extras.get("resource_df", pd.DataFrame())
    frc_plot: Optional[Path] = extras.get("frc_plot")
    drift_plot: Optional[Path] = extras.get("drift_plot")
    kpis: List[Tuple[str, str]] = extras.get("kpis", [])
    figures: List[Tuple[Path, str]] = extras.get("figures", [])
    runtime_captured: bool = extras.get("runtime_captured", False)

    runtime_df = normalize_runtime_units(runtime_df)

    def img(path: Optional[Path], cls: str = "plot", alt: str = "") -> str:
        uri = image_to_data_uri(path) if path else ""
        return f'<img class="{cls}" src="{uri}" alt="{alt}" />' if uri else ""

    # --- header ---
    input_text = summary.get("input", comp_summary.get("benchmark_dir", ""))
    header = (
        f"<h1>SMLM Pipeline Run Report</h1>"
        f'<p class="subtle"><strong>Generated:</strong> {created_at}</p>'
        f'<p class="subtle"><strong>Run folder:</strong> <code>{run_dir}</code></p>'
        f'<p class="subtle"><strong>Input:</strong> <code>{input_text}</code></p>'
        f'<p class="subtle"><strong>Backend:</strong> '
        f'<code>{summary.get("backend_name", "")}</code> &nbsp; '
        f'<strong>Host:</strong> <code>{comp_summary.get("hostname", "")}</code> &nbsp; '
        f'<strong>OS:</strong> <code>{comp_summary.get("os", "")}</code></p>'
    )

    # --- KPI grid ---
    kpi_cards = "".join(
        f'<div class="metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in kpis
    )
    kpi_grid = f'<div class="summary-grid">{kpi_cards}</div>' if kpi_cards else ""

    # --- TOC ---
    toc_items = [("system", "System"), ("runtime", "Runtime"),
                 ("resources", "Resources"), ("localization-qc", "Localization QC"),
                 ("resolution", "Resolution"), ("drift", "Drift"),
                 ("export", "Export"), ("figures", "Figures")]
    if not batch_df.empty:
        toc_items.insert(3, ("batches", "Batches"))
    toc = '<div class="toc">' + "".join(
        f'<a href="#{sid}">{name}</a>' for sid, name in toc_items
    ) + "</div>"

    sections: List[str] = []

    # System
    sections.append(
        _html_section("system", "System", make_html_table(machine_spec_rows(machine_specs)))
    )

    # Runtime
    runtime_cols = [
        "stage", "batch_index", "elapsed_sec", "elapsed_min", "status",
        "rss_mb", "process_cpu_percent", "gpu_peak_memory_allocated_mb",
    ]
    if runtime_captured and not runtime_df.empty:
        runtime_body = make_html_table(select_existing_columns(runtime_df, runtime_cols))
        runtime_body += img(runtime_plot, alt="Runtime plot")
    else:
        runtime_body = (
            '<p class="note">No per-stage timing was captured for this run '
            "(<code>runtime_benchmark.csv</code> is empty &mdash; psutil/torch were "
            "unavailable).</p>"
        )
    sections.append(_html_section("runtime", "Runtime", runtime_body))

    # Resources
    if not resource_df.empty:
        res_body = make_html_table(resource_df, max_rows=30)
    else:
        res_body = '<p class="note">No resource samples captured for this run.</p>'
    sections.append(_html_section("resources", "Resource usage", res_body))

    # Localization QC
    loc_body = make_html_table(single_row_kv(locqc_df, drop=("benchmark_layer",)))
    loc_body += img(loc_plot, alt="Localizations per movie")
    sections.append(_html_section("localization-qc", "Localization QC", loc_body))

    # Resolution + FRC
    res_cols = ["metric", "value", "unit", "status", "notes"]
    resolution_body = make_html_table(select_existing_columns(resolution_df, res_cols))
    if frc_plot is not None:
        resolution_body += "<h3>FRC curve</h3>" + img(frc_plot, alt="FRC curve")
    sections.append(_html_section("resolution", "Resolution", resolution_body))

    # Drift
    drift_summary_cols = [
        "status", "method", "max_abs_dx", "max_abs_dy", "max_radial_drift",
        "median_radial_drift", "p95_radial_drift",
        "linear_radial_drift_slope_per_frame", "message",
    ]
    drift_summary = select_existing_columns(drift_df.head(1), drift_summary_cols)
    drift_body = make_html_table(single_row_kv(drift_summary))
    drift_body += img(drift_plot, alt="Drift vs frame")
    sections.append(_html_section("drift", "Drift proxy", drift_body))

    # Export validation
    export_cols = ["export_name", "exists", "rows", "columns_ok", "status", "message"]
    sections.append(
        _html_section(
            "export", "Export validation",
            make_html_table(select_existing_columns(export_df, export_cols)),
        )
    )

    # Batches + QC previews
    if not batch_df.empty:
        batch_cols = [
            "batch_index", "input_name", "qc_status", "backend_status",
            "canonical_status", "shape", "axes", "dtype", "n_localizations",
        ]
        batch_body = make_html_table(select_existing_columns(batch_df, batch_cols))
        previews = collect_preview_cards(batch_df)
        if previews:
            batch_body += "<h3>QC previews</h3>" + previews
        sections.append(_html_section("batches", "Batch summary", batch_body))

    # Figures
    if figures:
        cards = "".join(
            f"<figure>{img(path, cls='', alt=caption)}"
            f"<figcaption>{caption}</figcaption></figure>"
            for path, caption in figures
        )
        sections.append(
            _html_section("figures", "Figures", f'<div class="gallery">{cards}</div>')
        )

    body = header + kpi_grid + toc + "\n".join(sections)

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>SMLM Pipeline Run Report</title>\n<style>"
        + HTML_CSS
        + "</style>\n</head>\n<body>\n"
        + body
        + "\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_run_report(run_dir: str | Path) -> Dict[str, str]:
    run_dir = Path(run_dir).expanduser().resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Summaries
    summary = read_json(run_dir / "run_summary.json")
    bench_summary = read_json(run_dir / "benchmark_summary.json")
    comp_summary = read_json(run_dir / "comparison_ready_summary.json")
    machine_specs = read_json(run_dir / "machine_specs.json") or bench_summary.get(
        "machine_specs", {}
    )

    # Metric layers
    batch_df = summarize_batches(run_dir)
    runtime_df = read_csv_safe(run_dir / "runtime_benchmark.csv")
    resource_df = read_csv_safe(run_dir / "resource_benchmark.csv")
    locqc_df = read_csv_safe(run_dir / "localization_qc_benchmark.csv")
    resolution_df = read_csv_safe(run_dir / "resolution_benchmark.csv")
    drift_df = read_csv_safe(run_dir / "drift_benchmark.csv")
    export_df = read_csv_safe(run_dir / "export_validation.csv")

    frc_files = sorted(run_dir.glob("frc_curve_batch_*.csv"))
    frc_df = read_csv_safe(frc_files[0]) if frc_files else pd.DataFrame()

    runtime_norm = normalize_runtime_units(runtime_df)
    runtime_captured = bool(
        not runtime_norm.empty
        and "elapsed_sec" in runtime_norm.columns
        and float(pd.to_numeric(runtime_norm["elapsed_sec"], errors="coerce").sum()) > 0
    )

    # Plots
    runtime_plot = make_runtime_plot(runtime_df, assets_dir / "runtime_by_stage.png")
    loc_plot = make_localization_plot(batch_df, assets_dir / "localizations_per_movie.png")
    frc_plot = make_frc_plot(frc_df, assets_dir / "frc_curve.png")
    drift_plot = make_drift_plot(drift_df, assets_dir / "drift_radial.png")

    # Figures: only run-specific figures, not static documentation/paper diagrams.
    figures = collect_run_figures(run_dir)

    kpis = build_kpis(bench_summary, comp_summary, locqc_df, runtime_captured)

    extras: Dict[str, Any] = {
        "bench_summary": bench_summary,
        "comp_summary": comp_summary,
        "machine_specs": machine_specs,
        "locqc_df": locqc_df,
        "resolution_df": resolution_df,
        "drift_df": drift_df,
        "export_df": export_df,
        "resource_df": resource_df,
        "frc_plot": frc_plot,
        "drift_plot": drift_plot,
        "kpis": kpis,
        "figures": figures,
        "runtime_captured": runtime_captured,
    }

    markdown = generate_markdown_report(
        run_dir, summary, batch_df, runtime_df, runtime_plot, loc_plot, extras
    )
    html = generate_html_report(
        run_dir, summary, batch_df, runtime_df, runtime_plot, loc_plot, extras
    )

    md_path = run_dir / "run_report.md"
    html_path = run_dir / "run_report.html"

    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    return {
        "markdown_report": str(md_path),
        "html_report": str(html_path),
        "assets_dir": str(assets_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SMLM pipeline run report.")
    parser.add_argument("--run", required=True, help="Pipeline run / benchmark directory.")
    args = parser.parse_args()

    outputs = generate_run_report(args.run)

    print("Report generated:")
    print(f"Markdown: {outputs['markdown_report']}")
    print(f"HTML:     {outputs['html_report']}")
    print(f"Assets:   {outputs['assets_dir']}")


if __name__ == "__main__":
    main()
