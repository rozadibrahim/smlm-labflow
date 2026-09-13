"""Run and check the standalone exporter on a synthetic localization table."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).with_name("localizations.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "outputs/export_demo")
    args = parser.parse_args()
    out = args.out.resolve()
    subprocess.run(
        [sys.executable, str(root / "export_downstream.py"),
         "--canonical", str(fixture), "--out", str(out)],
        check=True,
    )
    export_dir = out / "downstream_exports"
    report = json.loads((export_dir / "downstream_export_report.json").read_text())
    source = pd.read_csv(fixture)
    picasso = pd.read_csv(export_dir / "picasso_thunderstorm.csv")
    napari = pd.read_csv(export_dir / "napari_points.csv")
    checks = {
        "export_report_passed": report["status"] == "passed" and not report["errors"],
        "row_counts_preserved": len(source) == len(picasso) == len(napari) == 4,
        "picasso_coordinates_preserved_nm": all(
            picasso[f"{axis}_nm"].equals(source[axis]) for axis in ("x", "y", "z")
        ),
        "napari_coordinates_preserved_nm": all(
            napari[axis].equals(source[axis]) for axis in ("x", "y", "z")
        ),
        "napari_axis_order_zyx": list(napari.columns[:3]) == ["z", "y", "x"],
        "frames_preserved": picasso["frame"].equals(source["frame"])
            and napari["frame"].equals(source["frame"]),
        "photon_and_background_mapping": picasso["intensity_photon"].equals(source["photons"])
            and picasso["offset_photon"].equals(source["background"]),
        "documented_placeholder_defaults": bool(
            picasso["sigma_nm"].eq(120.0).all()
            and picasso["uncertainty_xy_nm"].eq(20.0).all()
        ),
    }
    result = {
        "scope": "Synthetic standalone export check; no LiteLoc inference or accuracy evaluation",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "exporter_sha256": hashlib.sha256((root / "export_downstream.py").read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "checks": checks,
        "passed": all(checks.values()),
    }
    (out / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Export checks failed; inspect checks.json")


if __name__ == "__main__":
    main()
