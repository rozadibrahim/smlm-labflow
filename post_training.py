#!/usr/bin/env python3
"""
post_training.py

Automatic post-training benchmark.

Whenever a LiteLoc training run passes, this script:

    1. Decodes `validLabels.pickle` — the held-out validation slice from the
       same simulator that LiteLoc was trained on (same PSF, same camera).
    2. Writes the simulated frames as a TIFF stack and the ground-truth
       emitter table (frame, x_nm, y_nm, z_nm, photons) as a CSV.
    3. Runs LiteLoc inference on those simulated frames using the
       just-trained checkpoint.
    4. Cross-matches predictions vs. truth with a greedy nearest-neighbour
       matcher within `match_radius_xy_nm` and `match_radius_z_nm`.
    5. Computes Jaccard index, RMSE_xy, RMSE_z, precision, recall, F1,
       bias, and Sage-style efficiency.
    6. Writes ONE row to `<train_dir>/benchmarks/post_training/
       post_training_benchmark.csv`.

This is the Sage-fight-club-style row that the rapport currently lacks.

Use as a CLI:

    python post_training.py --train-dir results/<run>/train

Or import and call:

    from post_training import run_post_training
    run_post_training(train_dir=Path("results/<run>/train"))
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import yaml

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

CHECKPOINT_CANDIDATES = ("checkpoint.pkl", "checkpoint.pt", "model.pkl")
VALIDLABELS_NAME = "validLabels.pickle"


def discover_train_artifacts(train_dir: Path) -> Dict[str, Path]:
    """Find the checkpoint, profile snapshot, and validLabels.pickle from a passed train run."""
    results_dir = train_dir / "results"
    registry_dir = train_dir / "registry"

    checkpoint: Optional[Path] = None
    for name in CHECKPOINT_CANDIDATES:
        candidate = results_dir / name
        if candidate.exists():
            checkpoint = candidate
            break
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found in {results_dir}. "
            f"Tried: {CHECKPOINT_CANDIDATES}"
        )

    valid_labels = results_dir / VALIDLABELS_NAME
    if not valid_labels.exists():
        raise FileNotFoundError(
            f"{VALIDLABELS_NAME} not found in {results_dir}. "
            "This script requires the held-out validation pickle that the "
            "LiteLoc trainer writes alongside the checkpoint."
        )

    profile_snapshot = registry_dir / "profile_snapshot.yaml"
    if not profile_snapshot.exists():
        # Try the resolved config as a fallback.
        alt = registry_dir / "resolved_config.yaml"
        if alt.exists():
            profile_snapshot = alt
        else:
            raise FileNotFoundError(
                f"Profile snapshot missing in {registry_dir}."
            )

    return {
        "checkpoint": checkpoint,
        "valid_labels": valid_labels,
        "profile_snapshot": profile_snapshot,
        "results_dir": results_dir,
        "registry_dir": registry_dir,
        "benchmarks_dir": train_dir / "benchmarks",
    }


# --------------------------------------------------------------------------- #
# Pickle decoding
# --------------------------------------------------------------------------- #

def _read_scaling_from_profile(profile_path: Path) -> Dict[str, float]:
    """
    Read pixel size, z scale, and photon scale from the LiteLoc profile snapshot.
    Falls back to the published DECODE-gateway defaults if a field is missing.
    """
    defaults = {
        "pixel_size_x_nm": 127.0,
        "pixel_size_y_nm": 117.0,
        "z_scale_nm": 960.0,
        "photon_scale": 15000.0,
    }
    if not profile_path.exists():
        return defaults
    try:
        cfg = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    out = dict(defaults)
    microscope = cfg.get("microscope") or {}
    if "pixel_size_x_nm" in microscope:
        out["pixel_size_x_nm"] = float(microscope["pixel_size_x_nm"])
    elif "pixel_size_nm" in microscope:
        out["pixel_size_x_nm"] = float(microscope["pixel_size_nm"])
    if "pixel_size_y_nm" in microscope:
        out["pixel_size_y_nm"] = float(microscope["pixel_size_y_nm"])

    train_block = (cfg.get("liteloc") or {}).get("runtime_yaml", {}).get("train", {})
    psf = train_block.get("PSF_model", {})
    if "z_scale" in psf:
        out["z_scale_nm"] = float(psf["z_scale"])
    training = train_block.get("Training", {})
    photon_range = training.get("photon_range")
    if isinstance(photon_range, (list, tuple)) and len(photon_range) >= 2:
        out["photon_scale"] = float(photon_range[1])
    return out


def decode_valid_labels(
    pkl_path: Path,
    scaling: Dict[str, float],
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Load `validLabels.pickle` and convert to (simulated stack, truth dataframe).

    LiteLoc serialises validation as
        dict[str_batch_id] = {
            "simu_image": Tensor[B, 1, H, W],
            "gt":         Tensor[B, N_max, 4]   columns (x_pix, y_pix, z_norm, intensity_norm)
            "s_mask":     Tensor[B, N_max]
            "locs", "x_os", "y_os", "z", "ints":  auxiliary training targets
        }

    We
        - stack all batches frame-by-frame,
        - convert (x_pix, y_pix) to (x_nm, y_nm) via the pixel sizes,
        - convert z_norm to z_nm via z_scale_nm,
        - convert intensity_norm to photons via photon_scale,
        - drop padded rows using `s_mask`.
    """
    # `torch` is needed to unpickle (the file contains torch.Tensor objects).
    try:
        import torch  # noqa: F401  (presence check; we convert to numpy below)
    except ImportError as exc:
        raise RuntimeError(
            "post_training.py needs `torch` to read validLabels.pickle. "
            "Run from inside the liteloc training env."
        ) from exc

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Unexpected validLabels structure: top-level is {type(obj).__name__}, "
            "expected dict."
        )

    pixel_size_x = scaling["pixel_size_x_nm"]
    pixel_size_y = scaling["pixel_size_y_nm"]
    z_scale = scaling["z_scale_nm"]
    photon_scale = scaling["photon_scale"]

    frames: List[np.ndarray] = []
    truth_rows: List[Dict[str, float]] = []
    frame_offset = 0
    batch_info: List[Dict[str, Any]] = []

    for batch_key in sorted(obj.keys(), key=lambda k: (len(str(k)), str(k))):
        batch = obj[batch_key]
        if not isinstance(batch, dict) or "simu_image" not in batch or "gt" not in batch:
            continue

        simu = batch["simu_image"].detach().cpu().numpy()  # [B, 1, H, W]
        gt = batch["gt"].detach().cpu().numpy()             # [B, N_max, 4]
        s_mask = batch["s_mask"].detach().cpu().numpy()     # [B, N_max]

        if simu.ndim == 4 and simu.shape[1] == 1:
            simu = simu[:, 0]  # [B, H, W]
        elif simu.ndim != 3:
            raise ValueError(
                f"Unexpected simu_image shape for batch {batch_key}: {simu.shape}"
            )

        batch_size = simu.shape[0]
        batch_info.append({
            "batch": str(batch_key),
            "n_frames": int(batch_size),
            "image_shape": list(simu.shape[1:]),
            "max_emitters_per_frame": int(gt.shape[1]),
        })
        frames.append(simu.astype(np.float32))

        for local_frame in range(batch_size):
            # Frame-level truth: extract emitters where the survival mask is 1.
            valid = s_mask[local_frame] > 0.5
            xs_pix = gt[local_frame, valid, 0]
            ys_pix = gt[local_frame, valid, 1]
            zs_norm = gt[local_frame, valid, 2]
            ints_norm = gt[local_frame, valid, 3]
            for x_pix, y_pix, z_norm, i_norm in zip(xs_pix, ys_pix, zs_norm, ints_norm):
                truth_rows.append({
                    "frame": int(frame_offset + local_frame + 1),  # 1-based
                    "x_nm": float(x_pix) * pixel_size_x,
                    "y_nm": float(y_pix) * pixel_size_y,
                    "z_nm": float(z_norm) * z_scale,
                    "photons": float(i_norm) * photon_scale,
                })

        frame_offset += batch_size

    if not frames:
        raise ValueError("No usable batches found in validLabels.pickle.")

    # Concatenate all batches into one stack so the existing inference path
    # treats them as a normal movie.
    stack = np.concatenate(frames, axis=0).astype(np.float32)
    truth_df = pd.DataFrame(truth_rows)

    metadata = {
        "n_batches": len(batch_info),
        "n_frames_total": int(stack.shape[0]),
        "n_truth_emitters": int(len(truth_df)),
        "image_height": int(stack.shape[1]),
        "image_width": int(stack.shape[2]),
        "pixel_size_x_nm": pixel_size_x,
        "pixel_size_y_nm": pixel_size_y,
        "z_scale_nm": z_scale,
        "photon_scale": photon_scale,
        "batches": batch_info,
    }
    return stack, truth_df, metadata


def save_simu_tiff(stack: np.ndarray, out_path: Path) -> None:
    """Write the simulated stack as a 16-bit TIFF (the format LiteLoc expects)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Clip then cast: simulator output already includes camera noise + offset.
    clipped = np.clip(stack, 0, np.iinfo(np.uint16).max)
    tifffile.imwrite(
        out_path,
        clipped.astype(np.uint16),
        photometric="minisblack",
        metadata={"axes": "TYX"},
    )


def save_truth_csv(truth: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(out_path, index=False)


# --------------------------------------------------------------------------- #
# Inference (delegates to run_pipeline.py infer)
# --------------------------------------------------------------------------- #

def invoke_liteloc_inference(
    *,
    project_root: Path,
    profile_path: Path,
    simu_tiff: Path,
    out_dir: Path,
    backend: str = "liteloc",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Run `python run_pipeline.py infer` on the simulated TIFF. Returns the
    canonical localizations CSV path. We use sys.executable so the call uses
    the same Python (and same env) that already finished training.
    """
    cmd = [
        sys.executable,
        str(project_root / "run_pipeline.py"),
        "infer",
        "-i", str(simu_tiff),
        "-p", str(profile_path),
        "-o", str(out_dir),
        "-b", backend,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[post_training] launching inference:\n  {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(project_root))

    # The infer command writes batches to <out_dir>/results/batches/. Find the
    # canonical localizations CSV.
    canonical_candidates = list(
        (out_dir / "results" / "batches").rglob("canonical_localizations.csv")
    )
    if not canonical_candidates:
        raise FileNotFoundError(
            f"Inference completed but no canonical_localizations.csv found in {out_dir}."
        )
    if len(canonical_candidates) > 1:
        # Take the largest (in case of leftover stale runs).
        canonical_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return canonical_candidates[0]


# --------------------------------------------------------------------------- #
# Matching and metric computation
# --------------------------------------------------------------------------- #

def greedy_match(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    match_radius_xy_nm: float,
    match_radius_z_nm: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Frame-by-frame greedy 1-NN matching with a hard xy radius and an optional
    hard z radius. Returns (pairs_df, counts_dict). Matches `benchmark.py`'s
    convention (Sage-fight-club style).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "post_training requires scipy.spatial.cKDTree for matching."
        ) from exc

    pred_local = pred.copy()
    truth_local = truth.copy()

    for df in (pred_local, truth_local):
        for col in ("frame", "x_nm", "y_nm", "z_nm"):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing for matching.")
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")

    pairs: List[Dict[str, float]] = []
    used_truth_ids: set[int] = set()

    common_frames = sorted(
        set(pred_local["frame"].dropna().astype(int).tolist())
        | set(truth_local["frame"].dropna().astype(int).tolist())
    )

    for frame in common_frames:
        p_frame = pred_local[pred_local["frame"] == frame]
        t_frame = truth_local[truth_local["frame"] == frame]
        if len(p_frame) == 0 or len(t_frame) == 0:
            continue

        p_xy = p_frame[["x_nm", "y_nm"]].to_numpy(dtype=float)
        t_xy = t_frame[["x_nm", "y_nm"]].to_numpy(dtype=float)
        tree = cKDTree(t_xy)
        distances, indices = tree.query(p_xy, k=1, distance_upper_bound=match_radius_xy_nm)

        p_indices = list(p_frame.index)
        t_indices = list(t_frame.index)

        for local_p, (dist, local_t) in enumerate(zip(distances, indices)):
            if np.isinf(dist) or local_t >= len(t_indices):
                continue
            t_id = int(t_indices[int(local_t)])
            if t_id in used_truth_ids:
                continue
            p_id = int(p_indices[local_p])
            dx = float(pred_local.loc[p_id, "x_nm"]) - float(truth_local.loc[t_id, "x_nm"])
            dy = float(pred_local.loc[p_id, "y_nm"]) - float(truth_local.loc[t_id, "y_nm"])
            dz_pred = pred_local.loc[p_id, "z_nm"] if "z_nm" in pred_local.columns else None
            dz_truth = truth_local.loc[t_id, "z_nm"] if "z_nm" in truth_local.columns else None
            dz = (
                float(dz_pred) - float(dz_truth)
                if dz_pred is not None and dz_truth is not None
                else None
            )
            if match_radius_z_nm is not None and dz is not None:
                if abs(dz) > match_radius_z_nm:
                    continue
            used_truth_ids.add(t_id)
            pairs.append({
                "frame": frame,
                "pred_index": p_id,
                "truth_index": t_id,
                "dx_nm": dx,
                "dy_nm": dy,
                "dz_nm": dz,
                "xy_distance_nm": float(dist),
            })

    tp = len(pairs)
    fp = int(len(pred_local) - tp)
    fn = int(len(truth_local) - tp)
    counts = {"tp": tp, "fp": fp, "fn": fn}
    return pd.DataFrame(pairs), counts


def compute_metrics(
    counts: Dict[str, int],
    pairs: pd.DataFrame,
    *,
    match_radius_xy_nm: float,
    match_radius_z_nm: Optional[float],
) -> Dict[str, Any]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else None

    rmse_xy = bias_x = bias_y = rmse_z = bias_z = None
    if tp > 0:
        dxs = pairs["dx_nm"].to_numpy(dtype=float)
        dys = pairs["dy_nm"].to_numpy(dtype=float)
        rmse_xy = float(np.sqrt(np.mean(dxs**2 + dys**2)))
        bias_x = float(np.mean(dxs))
        bias_y = float(np.mean(dys))
        if "dz_nm" in pairs.columns and pairs["dz_nm"].notna().any():
            dzs = pairs["dz_nm"].dropna().to_numpy(dtype=float)
            if dzs.size:
                rmse_z = float(np.sqrt(np.mean(dzs**2)))
                bias_z = float(np.mean(dzs))

    # Sage-fight-club efficiency: sqrt(Jaccard * (1 - RMSE_xy / R_xy)),
    # clipped so it stays in [0, 1].
    efficiency = None
    if jaccard is not None and rmse_xy is not None and match_radius_xy_nm > 0:
        norm_term = max(0.0, 1.0 - rmse_xy / float(match_radius_xy_nm))
        efficiency = float(np.sqrt(jaccard * norm_term))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "rmse_xy_nm": rmse_xy,
        "rmse_z_nm": rmse_z,
        "bias_x_nm": bias_x,
        "bias_y_nm": bias_y,
        "bias_z_nm": bias_z,
        "efficiency_sage": efficiency,
        "match_radius_xy_nm": match_radius_xy_nm,
        "match_radius_z_nm": match_radius_z_nm,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_post_training(
    train_dir: Path,
    *,
    project_root: Optional[Path] = None,
    backend: str = "liteloc",
    match_radius_xy_nm: float = 250.0,
    match_radius_z_nm: Optional[float] = 500.0,
    skip_inference: bool = False,
    extra_infer_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full post-training benchmark. Returns the metric dictionary that
    is also written to disk as one row of the benchmark CSV.

    `match_radius_xy_nm` defaults to 250 nm — Sage et al.'s "lenient" radius
    so a mediocre run still produces non-zero scores. The user can tighten it
    via the CLI.
    """
    train_dir = Path(train_dir).expanduser().resolve()
    project_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parent
    )

    artifacts = discover_train_artifacts(train_dir)
    scaling = _read_scaling_from_profile(artifacts["profile_snapshot"])

    out_dir = artifacts["benchmarks_dir"] / "post_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[post_training] train_dir         = {train_dir}")
    print(f"[post_training] checkpoint        = {artifacts['checkpoint'].name}")
    print(f"[post_training] valid labels      = {artifacts['valid_labels'].name}")
    print(f"[post_training] profile snapshot  = {artifacts['profile_snapshot'].name}")
    print(f"[post_training] scaling           = {scaling}")

    # 1 + 2. Decode pickle, write TIFF + truth CSV.
    stack, truth_df, decode_meta = decode_valid_labels(
        artifacts["valid_labels"], scaling
    )
    simu_tiff = out_dir / "simu_validation.tif"
    truth_csv = out_dir / "truth_validation.csv"
    save_simu_tiff(stack, simu_tiff)
    save_truth_csv(truth_df, truth_csv)
    print(
        f"[post_training] wrote {decode_meta['n_frames_total']} simulated frames "
        f"({decode_meta['image_height']}x{decode_meta['image_width']}) to "
        f"{simu_tiff.name}, {decode_meta['n_truth_emitters']} truth emitters to "
        f"{truth_csv.name}"
    )

    canonical_pred_csv: Optional[Path] = None
    inference_status = "skipped"
    # The run_pipeline.py resolver locates the shared registry at <run_parent>/../registry/.
    # If we nest the inference dir deep inside benchmarks/, the resolver can't find it.
    # Place inference output as a SIBLING of the train run so it shares the same global
    # registry that already points at the trained checkpoint + calibration.
    inference_dir = train_dir.parent / f"{train_dir.name}_post_training_infer"
    if not skip_inference:
        # 3. Run inference. Subprocess so it uses the same env that just trained.
        # Note: a non-zero exit code does NOT necessarily mean inference failed.
        # On Windows the post-inference plotting path can crash with a fatal
        # MKL DLL exception AFTER the canonical CSV has already been written.
        # We therefore probe for the CSV on disk regardless of the exit code.
        inference_dir.mkdir(parents=True, exist_ok=True)
        try:
            canonical_pred_csv = invoke_liteloc_inference(
                project_root=project_root,
                profile_path=artifacts["profile_snapshot"],
                simu_tiff=simu_tiff,
                out_dir=inference_dir,
                backend=backend,
                extra_args=extra_infer_args,
            )
            inference_status = "passed"
        except subprocess.CalledProcessError as exc:
            # Look for the canonical CSV on disk - it may have been written
            # before the post-inference report path crashed.
            candidates = list(
                (inference_dir / "results" / "batches").rglob("canonical_localizations.csv")
            )
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
                canonical_pred_csv = candidates[0]
                inference_status = (
                    f"passed (post-inference crashed: exit {exc.returncode}; "
                    "canonical CSV was written before the crash)"
                )
                print(
                    f"[post_training] subprocess exited {exc.returncode} "
                    f"but canonical CSV was produced at {canonical_pred_csv}",
                    file=sys.stderr,
                )
            else:
                inference_status = f"failed (exit {exc.returncode})"
                print(f"[post_training] inference failed: {exc}", file=sys.stderr)
        except FileNotFoundError as exc:
            inference_status = "failed (no canonical CSV)"
            print(f"[post_training] inference output missing: {exc}", file=sys.stderr)

    # 4 + 5. Match and compute metrics.
    pairs_df = pd.DataFrame()
    metrics: Dict[str, Any] = {}
    n_pred = 0
    if canonical_pred_csv is not None and canonical_pred_csv.exists():
        pred_df = pd.read_csv(canonical_pred_csv)
        rename_map = {}
        if "x_nm" not in pred_df.columns and "x" in pred_df.columns:
            rename_map["x"] = "x_nm"
        if "y_nm" not in pred_df.columns and "y" in pred_df.columns:
            rename_map["y"] = "y_nm"
        if "z_nm" not in pred_df.columns and "z" in pred_df.columns:
            rename_map["z"] = "z_nm"
        if rename_map:
            pred_df = pred_df.rename(columns=rename_map)

        n_pred = int(len(pred_df))
        pairs_df, counts = greedy_match(
            pred_df,
            truth_df,
            match_radius_xy_nm=match_radius_xy_nm,
            match_radius_z_nm=match_radius_z_nm,
        )
        metrics = compute_metrics(
            counts,
            pairs_df,
            match_radius_xy_nm=match_radius_xy_nm,
            match_radius_z_nm=match_radius_z_nm,
        )
        pairs_df.to_csv(out_dir / "match_pairs.csv", index=False)
    else:
        metrics = {
            "tp": None, "fp": None, "fn": None,
            "precision": None, "recall": None, "f1": None, "jaccard": None,
            "rmse_xy_nm": None, "rmse_z_nm": None,
            "bias_x_nm": None, "bias_y_nm": None, "bias_z_nm": None,
            "efficiency_sage": None,
            "match_radius_xy_nm": match_radius_xy_nm,
            "match_radius_z_nm": match_radius_z_nm,
        }

    # 6. Write benchmark CSV (append-friendly: one row per run).
    row = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_dir": str(train_dir),
        "backend": backend,
        "checkpoint": str(artifacts["checkpoint"]),
        "valid_labels": str(artifacts["valid_labels"]),
        "n_truth_frames": decode_meta["n_frames_total"],
        "n_truth_emitters": decode_meta["n_truth_emitters"],
        "n_predictions": n_pred,
        "inference_status": inference_status,
        **metrics,
    }
    bench_csv = out_dir / "post_training_benchmark.csv"
    if bench_csv.exists():
        prev = pd.read_csv(bench_csv)
        combined = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined.to_csv(bench_csv, index=False)

    bench_json = out_dir / "post_training_benchmark.json"
    bench_json.write_text(
        json.dumps({"row": row, "decode_meta": decode_meta}, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"[post_training] wrote benchmark row to {bench_csv}")
    print(f"[post_training] precision={row.get('precision')}  "
          f"recall={row.get('recall')}  f1={row.get('f1')}  "
          f"jaccard={row.get('jaccard')}  rmse_xy={row.get('rmse_xy_nm')}  "
          f"rmse_z={row.get('rmse_z_nm')}")

    return row


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the post-training Sage-style benchmark."
    )
    p.add_argument(
        "--train-dir",
        required=True,
        type=Path,
        help="Path to a finished `train` run folder (containing results/, registry/, benchmarks/).",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="smlm-labflow project root (defaults to the script's directory).",
    )
    p.add_argument(
        "--backend",
        default="liteloc",
        help="Backend to use for inference (default: liteloc).",
    )
    p.add_argument(
        "--match-radius-xy-nm",
        type=float,
        default=250.0,
        help="xy match radius in nm (default: 250).",
    )
    p.add_argument(
        "--match-radius-z-nm",
        type=float,
        default=500.0,
        help="z match radius in nm (default: 500; pass 0 to disable z gating).",
    )
    p.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only decode the validation pickle and write TIFF + truth CSV; do not run inference.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    z_radius: Optional[float] = (
        None if args.match_radius_z_nm <= 0 else float(args.match_radius_z_nm)
    )
    try:
        run_post_training(
            train_dir=args.train_dir,
            project_root=args.project_root,
            backend=args.backend,
            match_radius_xy_nm=args.match_radius_xy_nm,
            match_radius_z_nm=z_radius,
            skip_inference=args.skip_inference,
        )
    except Exception as exc:
        print(f"[post_training] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
