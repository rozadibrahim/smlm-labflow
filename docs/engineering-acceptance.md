# Engineering acceptance — 2026-09-13

**The installed subset passes the checks below. The full advertised toolkit is
not yet approved for release.** The registry contains 48 scientific method entries
and two mechanism tests; 15 scientific adapters still have unfinished bindings.

## Measured results

- **42 tests passed locally and 42 on the RunPod.** Checks cover file contracts,
  environment isolation, arbitrary working directories, failed-install receipts,
  provenance, numerical fixtures and clear CLI errors.
- **31 conformance checks passed, zero failed, 19 skipped.** Of the passes, 30 are
  scientific method adapters and one is a local subprocess mechanism test.
  Required segmenters, AIM, RCC, trackpy, XGBoost and LightGBM were explicitly gated.
- **9/9 demo stages produced outputs.** No microscope dataset was used.
- A **clean core bootstrap** into a new environment succeeded and passed its
  available test suite. The main working core remains isolated from all GPU envs.
- **LiteLoc upstream network checks passed separately:** imports, CPU/CUDA forward
  agreement, postprocessing shape/finite values, backward plus NAdam step, and a
  state-dict checkpoint round-trip. These use random weights and tiny generated
  frames; they do not claim full trained-model inference, calibration or scientific
  accuracy. The generic localize conformance stage therefore still reports SKIP.
- Cellpose, StarDist, micro-SAM and Omnipose ran real pretrained-model inference
  through LabFlow. Each returned a 128×128 uint32 label TIFF; object counts were
  3, 2, 3 and 2 respectively. These are integration fixtures, not accuracy scores.
  A separate Omnipose run with `--param gpu=true` also completed successfully.

## Tested installation

The live Pod uses an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
The core uses Python 3.12; LiteLoc uses Python 3.9 and Torch 2.7.1+cu128.
Cellpose 3.1.1.2, micro-SAM 1.8.14 and Omnipose 1.1.4 each have a separate
CUDA 12.8 Torch 2.7.1 environment. StarDist 0.9.2 / CSBDeep 0.8.2 run in an
isolated TensorFlow CPU 2.16.2 environment. Julia 1.10.12 runs the AIM port.

The Omnipose recipe pins ncolor 1.4.3: the newer resolved release removed
`unique_nonzero`, causing a real import failure. The installer now refuses to
record success after such an import failure. Package inventories were saved for
all installed environments; top-level version pins alone are not complete locks.

Source and artifacts are on persistent storage:

- Checkout: `/workspace/smlm-labflow`.
- LiteLoc source: `/workspace/backends/LiteLoc`, upstream commit
  `aa4d262b89c7a872debaf908d82e1aaa26fe6915`.
- Model caches: `/workspace/models`.
- Evidence: `/workspace/outputs/engineering-validation` — `pytest.log`,
  `pytest-clean.log`, `bootstrap-clean.log`, `conformance/conformance.json`,
  per-method outputs/provenance, `liteloc-network.json`, and `package-inventories/`.
- Executable environments: `/opt/labflow-envs`; these must be recreated if the
  container's local filesystem is replaced. Startup performs no installation.

These changes were installed on the live Pod from the feature checkout. They
are **not baked into the existing GHCR startup image**, and master remains unchanged.

## Engineering defects addressed

1. Installation and execution now resolve the same absolute venv path. Backend
   Python cannot silently fall back to the core interpreter, and inherited core
   `PYTHONPATH`/`PYTHONHOME` are removed for isolated commands.
2. A venv directory is no longer treated as a completed installation. Receipts
   require dependency/import checks, include the requirements content hash, and
   are invalidated before a reinstall so failure cannot retain a success marker.
3. The four segmenters have native recipes; they no longer require nested Docker
   on the Pod. LiteLoc has a pinned source installer and explicit interpreter routing.
4. Truncated, floating-point or negative-label segmentation TIFFs are rejected.
   Segmentation runners use uint32 labels instead of silently wrapping at 65,535.
5. Conformance preserves separate output directories for each method, including
   sidecars. `--require` treats missing, skipped or failing tools as failed acceptance.
6. DME uncertainty mapping now passes standard deviations in pixels rather than
   squaring them before an API that squares internally. Pixel-mode z/uncertainty
   conversion is consistent, and plotting is disabled for headless execution.
   This mapping has mocked upstream-boundary tests; DME's native execution remains
   blocked. [Upstream native implementation](https://github.com/qnano/drift-estimation/blob/2ee39f74ac5ac03daee61af39f37a75051c56fa3/dme/DME/DriftEstimation.cu)
7. Core Python metadata now states the actual >=3.11 requirement and bootstrap
   stops before installing with an incompatible interpreter.
8. README and installation docs no longer imply that dependency installation
   completes unfinished adapters or that unverified distribution routes are ready.

## Remaining acceptance gates

- Implement the 15 unfinished adapters listed below against identifiable upstream
  versions. Do not substitute generic algorithms under research-tool names.
- Supply a real DECODE environment recipe and verify its currently speculative
  inference binding against the selected upstream API.
- Complete DME's Linux native build. Both an unmodified CPU build and a build with
  the missing math header supplied failed; compiler logs are retained. The upstream
  README also identifies CPU-only compilation as problematic. No broken native
  package was installed into the core. [Upstream build notes](https://github.com/qnano/drift-estimation#build-from-source-tested-on-windows-and-linux)
- Add small reference model/calibration fixtures for the full LiteLoc lifecycle
  and coordinate export, including legacy checkpoint loading under newer Torch.
  A random-weight network check is not a substitute for that adapter boundary test.
- Verify units, frame indexing, axis conventions, empty outputs, model provenance,
  multi-file dependencies, recovery and cancellation for each completed backend.
- Validate desktop GUI behavior and a fresh complete LiteLoc environment from its
  new installer; this session validated the existing-image environment reuse path.
- Generate full platform locks, perform clean-machine acceptance, then rebuild and
  validate the distributable image/package. Do not merge to master until approved.

This is engineering validation, not a repeat of the papers' benchmarks. Small
known-output/reference fixtures are still needed to show that LabFlow preserves
upstream behavior. Some in-core methods are ports or reimplementations, so they
also require their own numerical checks rather than inheriting a paper's result.

## Per-method conformance

| Method | Stage | Runtime | Result | Detail |
|---|---|---|---|---|
| liteloc | localize | external | SKIP | Generic stage SKIP; separate upstream network engineering checks PASS. |
| decode | localize | conda | SKIP | no engineering fixture configured (model/calibration may be required) |
| fd_deeploc | localize | conda | SKIP | adapter not implemented |
| deepstorm3d | localize | conda | SKIP | adapter not implemented |
| none | drift | python | PASS |  |
| rcc | drift | python | PASS |  |
| fiducial | drift | python | PASS |  |
| aim_julia | drift | python | PASS |  |
| dme | drift | python | SKIP | not installed (needs its env) |
| render | render | python | PASS |  |
| cellpose | segment | venv | PASS |  |
| stardist | segment | venv | PASS |  |
| microsam | segment | venv | PASS |  |
| omnipose | segment | venv | PASS |  |
| magik | track | docker | SKIP | adapter not implemented |
| trackmate | track | docker | SKIP | adapter not implemented |
| spot_on | track | venv | SKIP | adapter not implemented |
| trackpy | track | python | PASS |  |
| swift | track | local | SKIP | adapter not implemented |
| dbscan | cluster | python | PASS |  |
| optics | cluster | python | PASS |  |
| hdbscan | cluster | python | PASS |  |
| locan | cluster | python | PASS |  |
| miro | cluster | venv | SKIP | adapter not implemented |
| srtesseler | cluster | python | PASS |  |
| caml | cluster | venv | SKIP | adapter not implemented |
| bayesian | cluster | venv | SKIP | adapter not implemented |
| ripley | spatial_stats | python | PASS |  |
| paircorrelation | spatial_stats | python | PASS |  |
| nnd | spatial_stats | python | PASS |  |
| voronoi | spatial_stats | python | PASS |  |
| gfunction | spatial_stats | python | PASS |  |
| cbc | spatial_stats | python | PASS |  |
| crosscorrelation | spatial_stats | python | PASS |  |
| qpaint | counting | python | PASS |  |
| blink | counting | python | PASS |  |
| ibfcs | counting | python | SKIP | adapter not implemented |
| clusternet | phenotype | venv | SKIP | adapter not implemented |
| msd | analyze | python | PASS |  |
| deeptrace | analyze | docker | SKIP | adapter not implemented |
| vbspt | analyze | venv | SKIP | adapter not implemented |
| andi | analyze | venv | SKIP | adapter not implemented |
| deepspt | analyze | docker | SKIP | adapter not implemented |
| frc | metrics | python | PASS |  |
| nena | metrics | python | PASS |  |
| randomforest | qc_audit | python | PASS |  |
| xgboost | qc_audit | python | PASS |  |
| lightgbm | qc_audit | python | PASS |  |
| _selftest_local | test | local | PASS |  |
| _docker_selftest | test | docker | SKIP | not installed (needs its env) |
