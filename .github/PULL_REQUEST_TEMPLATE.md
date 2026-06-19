## What this changes

<!-- one or two sentences -->

## Checklist
- [ ] `labflow doctor` is clean (registry lints)
- [ ] `labflow conformance` passes (or the new method SKIPs cleanly when its env isn't installed)
- [ ] `pytest labflow/tests driftcorr/tests` passes
- [ ] If adding a method: one entry in `config/methods.yaml`; in-core methods plumb files via `labflow.io`
- [ ] No fabricated tool APIs — research-grade calls left as clear binding points
- [ ] ASCII only in CLI/`print` strings

## Notes for reviewers
<!-- anything that can't run in CI (GPU/Fiji/Julia), binding points left open, etc. -->
