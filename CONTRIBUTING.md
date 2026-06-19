# Contributing to SMLM LabFlow

Thanks for helping! The architecture is built so that **adding a method is a single
registry entry** — most contributions don't touch the core.

## Add a method (the common case)
1. Declare it once in [`config/methods.yaml`](config/methods.yaml).
2. For an **in-core** (light, pure-Python) method, copy
   [`labflow/stages/_template.py`](labflow/stages/_template.py) and write the science;
   plumb files through `labflow.io` (`read_localizations` / `write_table`).
   For a **heavy/conflicting** tool, point the entry at an isolated `venv`/`conda`/`docker`
   runtime — see [`docs/methods.md`](docs/methods.md) and [`docs/environments.md`](docs/environments.md).
3. Verify:
   ```bash
   labflow doctor          # lints your registry entry
   labflow conformance     # runs every installed method end-to-end on synthetic data
   pytest labflow/tests driftcorr/tests
   ```
4. Open a PR. CI ([test.yml](.github/workflows/test.yml)) runs the lint + tests + conformance.

Don't fabricate a tool's API: if a model/CLI isn't documented, leave a clear **binding
point** (a `NotImplementedError` with guidance) and wire the IO + file contract around it —
several DL backends are integrated this way.

## Dev setup
```bash
python bootstrap.py --extras light,dev      # or: pip install -e ".[light,dev]"
```

## Conventions
- ASCII in CLI/`print` strings (Windows cp1252).
- Keep the core light; heavy deps go in a per-tool env, never the core.
- One method = one YAML entry that appears in the CLI **and** both pipeline engines.

## Reporting bugs / ideas
Use the issue templates. For security-sensitive reports, email the maintainer rather than
opening a public issue.
