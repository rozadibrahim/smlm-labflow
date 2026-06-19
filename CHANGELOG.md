# Changelog

All notable changes to SMLM LabFlow. Format: [Keep a Changelog](https://keepachangelog.com);
versioning: [SemVer](https://semver.org).

## [Unreleased]

## [0.3.0] - 2026-06
### Added
- **Registry-driven spine**: one `config/methods.yaml` drives the `labflow` CLI **and** two
  workflow engines (Snakemake + Nextflow). ~50 methods across 13 stages.
- **In-core backends** (run end-to-end, verified by `labflow conformance`): drift
  (none/rcc/fiducial/aim), cluster (dbscan/optics/hdbscan/locan/srtesseler), spatial_stats
  (ripley/paircorrelation/nnd/voronoi/gfunction/cbc/crosscorrelation), counting
  (qpaint/blink), analyze (msd), metrics (frc/nena), render, qc_audit (randomforest).
- **Heavy DL tools wired** with isolated envs/images + honest binding points: DECODE,
  FD-DeepLoc, DeepSTORM3D, Cellpose, StarDist, micro-SAM, Omnipose, MAGIK, TrackMate,
  Spot-On, swift, MIRO, CAML, Bayesian, DeepTRACE, DeepSPT, vbSPT, AnDi, ClusterNet.
- **Isolation & reproducibility**: per-tool runtimes, file-contract validation, provenance
  manifests, `labflow doctor`, the conformance harness, `labflow install`, `bootstrap.py`.
- **Distribution**: conda-forge recipe, `pixi` cross-platform locks, a napari plugin,
  nf-core-style institutional configs, two-path [install docs](docs/install.md),
  `labflow demo` (synthetic end-to-end run), `CITATION.cff`, a JOSS paper draft.
- **CI**: unit tests + conformance on every push; tool-image builds on release.

### Changed
- License set to **MIT** (was unset).

### Notes
- Heavy DL tools are *wired*, not yet GPU-validated; several have explicit binding points.
- Picasso, ThunderSTORM, and SMAP are intentionally not integrated.
