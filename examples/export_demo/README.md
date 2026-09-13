# Standalone export example

This example runs `export_downstream.py` on a four-row synthetic localization table. It was added to make the export behavior inspectable without installing LiteLoc or downloading microscopy data. It is not an experimental dataset or an internship result.

## Run

From the repository root, with Python and pandas installed:

```bash
python -m pip install -r examples/export_demo/requirements.txt
python examples/export_demo/run.py
```

The script invokes the actual standalone export CLI, reads its output files, and exits with an error if a check fails. Generated files go to `outputs/export_demo/` by default; use `--out /path/to/folder` to choose another location.

## Input

[localizations.csv](localizations.csv) contains four hand-specified synthetic points over three frames. All coordinates are in nanometers; frames are one-based. Asymmetric x/y coordinates and both negative and positive z values make axis-order mistakes easier to see. Photon, background, and confidence values are synthetic too.

| Frame | x (nm) | y (nm) | z (nm) |
|---|---|---|---|
| 1 | 110 | 330 | -50 |
| 1 | 440 | 220 | 0 |
| 2 | 115 | 335 | 50 |
| 3 | 880 | 660 | 100 |

## Recorded result

Run on 13 September 2026 using Python 3.12.14 and pandas 2.2.3. The exporter source was taken from LabFlow commit `15b69cf2714bcc419835b5f1a0d866f56d9e890e`.

- [checks.json](recorded/checks.json): eight checks passed, with source and input SHA-256 hashes.
- [picasso_thunderstorm.csv](recorded/picasso_thunderstorm.csv): four rows; x/y/z values preserved as x_nm/y_nm/z_nm.
- [napari_points.csv](recorded/napari_points.csv): four rows; coordinate columns ordered z, y, x, with frame and other properties retained.

The checks cover row counts, coordinate values, axis order, frame values, photon/background mapping, export status, and documented defaults. The recorded files were copied from the run; rerunning writes fresh files under the output directory.

## What the example does not establish

No localization model is run, so there is no localization error, resolution, or speed comparison. Neither Picasso nor napari was opened to validate imports. This example also does not exercise the separate exports in `post_inference.py`.

The standalone exporter assumes coordinates are already in nanometers. Its `sigma_nm=120` and `uncertainty_xy_nm=20` values are placeholders from the exporter defaults, not estimates from these points. Check or replace such values before scientific use.
