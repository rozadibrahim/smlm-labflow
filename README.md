# SMLM LabFlow

A Python workflow for **Single-Molecule Localization Microscopy (SMLM)**, developed as part of a short M1 research internship.

**Status: work in progress.** The current localization backend is LiteLoc.

## Internship context and project contribution

This project organizes the steps around SMLM localization: preparing a run, passing calibration and training settings to LiteLoc, processing localization tables, and collecting QC and output files.

LiteLoc provides the localization model and its training/inference implementation. LabFlow's contribution is the workflow around that backend:

- A command-line interface for calibration, training, and inference.
- Scientific profiles, machine-specific backend paths, and a registry for reusing calibration/model artifacts.
- Input QC, conversion to a common localization table, and downstream exports.
- Benchmarking and run-reporting code.

The code is linked below so each part can be inspected. The [export example](examples/export_demo/README.md) is a reproducible check of one component. An experimental run with its dataset, configuration, and results still needs to be documented here.

## Code map

| Part | Implementation |
|---|---|
| Pipeline entry point and profile inheritance | [run_pipeline.py](run_pipeline.py) |
| Configuration and artifact resolution | [adapters/resolver.py](adapters/resolver.py) |
| Calls into LiteLoc | [adapters/liteloc_adapter.py](adapters/liteloc_adapter.py) |
| TIFF input QC | [qc_input.py](qc_input.py) |
| Localization conversion, QC, and exports | [post_inference.py](post_inference.py) |
| Standalone canonical CSV exporter | [export_downstream.py](export_downstream.py) |
| Benchmarking and scientific metrics | [benchmark.py](benchmark.py), [quality_metrics.py](quality_metrics.py) |
| Run reports | [generate_run_report.py](generate_run_report.py) |
| Interactive review | [napari_locan_review.py](napari_locan_review.py) |

## Try a small example

The [CPU export example](examples/export_demo/README.md) uses four explicitly synthetic localizations. It runs the existing standalone exporter, saves both output tables, and checks their contents. It needs pandas; LiteLoc, microscopy data, and a GPU are not required.

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r examples/export_demo/requirements.txt
python examples/export_demo/run.py
```

See the [recorded checks](examples/export_demo/recorded/checks.json) and [sample output](examples/export_demo/recorded/picasso_thunderstorm.csv). These establish the behavior of the standalone export path on the fixture, not localization accuracy or compatibility with every downstream application.

## Run with LiteLoc

The following commands are setup examples for your own data. The full calibration/training/inference sequence has not been rerun as part of the export example.

### Environment and backend

```bash
conda env create -f env_yamls/liteloc_env_base.yml
conda activate liteloc_env
```

Install LiteLoc separately and configure `adapters/backend_paths.yml` using [the example](adapters/backend_paths.example.yml). Set `liteloc.root` to its installation directory and check that the module/function mappings match your LiteLoc revision.

The base environment specifies Python 3.9 and PyTorch with CUDA 12.1. Training and inference in the example profile request CUDA. Use a compatible GPU environment for those stages.

### Scientific profile

Copy [profiles/liteloc_unified_example.yaml](profiles/liteloc_unified_example.yaml) and adapt the microscope, camera, PSF, and training settings. The supplied values are examples, not a calibration for your instrument.

Machine paths belong in `adapters/backend_paths.yml`; acquisition and analysis settings belong in the profile. Fields set to `auto` are resolved from the run inputs or available registry artifacts. Check the resolved configuration before relying on a reused model or calibration.

### Commands

Replace the input paths with your data and the profile path with your edited copy. The sibling output folders allow the stages to use the shared registry under `outputs/registry/`.

```bash
python run_pipeline.py calibrate \
  -i /data/beads \
  -p profiles/liteloc_unified_example.yaml \
  -o outputs/calibration -b liteloc

python run_pipeline.py train \
  -i /data/training_frames \
  -p profiles/liteloc_unified_example.yaml \
  -o outputs/training -b liteloc

python run_pipeline.py infer \
  -i /data/raw_movies \
  -p profiles/liteloc_unified_example.yaml \
  -o outputs/inference -b liteloc \
  --export generic --export picasso --export napari
```

The inference CLI defaults to raw backend output. Select exports explicitly; available choices include `raw`, `generic`, `smap`, `picasso`, `napari`, and `locan`. Add `--dry-run` to preview a stage or `--max-files 1` to limit an inference run.

A run organizes artifacts under `results/`, `benchmarks/`, `reports/`, and `registry/`. Inspect stage status, logs, and resolved configuration as well as the localization CSV.

## Design choices and tradeoffs

- **Separate paths from scientific settings.** This keeps a microscope profile reusable when LiteLoc is installed somewhere else. The module mappings still depend on the backend version.
- **Convert to a common table.** A shared schema gives QC and exports a consistent input. Coordinate units, frame conventions, and missing uncertainty fields still need attention.
- **Reuse artifacts through a registry.** This reduces manual path entry between stages. A previous artifact can belong to a different experimental condition, so compatibility must be checked.
- **Keep inspection outputs.** QC tables, plots, and run reports provide intermediate evidence to examine when a run behaves unexpectedly. A successful software status alone does not establish scientific validity.

## Validation and limitations

- The committed [synthetic example](examples/export_demo/README.md) checks the standalone CSV export path only.
- No experimental accuracy or runtime result is claimed by that example. Use the [experimental run guide](docs/experimental_run.md) to document a measured result with its context.
- The standalone `export_downstream.py` assumes coordinates are already in nanometers. The broader `post_inference.py` has separate unit-conversion logic; its `auto` mode uses a heuristic. Set units and pixel size explicitly when known.
- Exported precision or background fields can contain fallback values. In the standalone example, `sigma_nm=120` and `uncertainty_xy_nm=20` are defaults, not fitted values.
- The standalone exporter writes a ThunderSTORM-style CSV for Picasso conversion; the main post-inference path has a different Picasso export. Neither file generation nor this example verifies import into the application.
- LiteLoc must be installed separately. Its revision, dependencies, microscope settings, and model/calibration artifacts affect reproducibility.
- Additional localization backends are planned; only the LiteLoc adapter is currently included.

## Next steps

- Document one experimental run with shareable input data, configuration, output, and interpretation.
- Validate downstream imports and coordinate conventions on that run.
- Improve CRLB/RMSE reporting, PSF diagnostics, and registry compatibility checks.
- Add profile examples and further backend adapters as they are tested.

## License

MIT — see [LICENSE](LICENSE). External tools, including LiteLoc, retain their own licenses.

## Citation

If you use this pipeline, please cite:

1. **SMLM LabFlow** (this repository)
2. **LiteLoc** — the backend localization engine
3. Any **downstream analysis tools** applied to results

### LiteLoc

> Li, Y. et al. *Scalable and lightweight deep learning for efficient high accuracy single-molecule localization microscopy.* **Nature Communications** (2025). https://doi.org/10.1038/s41467-025-62662-5

```bibtex
@article{li2025liteloc,
  author  = {Li, Yue and others},
  title   = {Scalable and lightweight deep learning for efficient high accuracy single-molecule localization microscopy},
  journal = {Nature Communications},
  year    = {2025},
  doi     = {10.1038/s41467-025-62662-5},
  url     = {https://www.nature.com/articles/s41467-025-62662-5}
}
```

### SMLM LabFlow

```bibtex
@software{smlm_labflow_2026,
  author  = {Ibrahim, Rozad},
  title   = {SMLM LabFlow: a modular wrapper pipeline for Single-Molecule Localization Microscopy workflows},
  year    = {2026},
  url     = {https://github.com/rozadibrahim/smlm-labflow}
}
```

---

## Maintainer

**Rozad Ibrahim** — ESBS, University of Strasbourg
