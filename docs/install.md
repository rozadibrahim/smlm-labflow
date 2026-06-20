# Installing SMLM LabFlow

There are **two front doors** — pick by who you are. They install the *same* tool; the
biologist path is turnkey and locked, the developer path gives full control. Neither
removes the other.

---

## A. Biologists / lab users (no code)

Pick whichever your lab already uses.

### conda / mamba (recommended)
```bash
mamba install -c conda-forge smlm-labflow      # once published to conda-forge
labflow doctor                                 # check your install
labflow conformance                            # prove the in-core backends run here
```

### pixi (locked, reproducible on any OS, works offline)
A committed `pixi.lock` pins every dependency bit-for-bit on Windows/macOS/Linux:
```bash
pixi install -e light          # build the locked environment
pixi run -e light labflow conformance
pixi shell -e gui              # then run the napari GUI below
```
For an offline microscope PC, ship the solved env with `pixi-pack` (no internet needed
on the target).

### napari (GUI — no command line)
```bash
pip install "smlm-labflow[gui]"      # or: mamba install -c conda-forge smlm-labflow napari
napari                               # Plugins menu -> "SMLM LabFlow: Open LabFlow run"
```
Point it at a run folder (or a localizations/clusters CSV) to view it as napari layers
(clusters coloured by `cluster_id`).

### Getting a heavy tool (Cellpose, DECODE, …)
Still one line — it pulls a prebuilt, isolated image; nothing is compiled on your machine:
```bash
labflow install cellpose
labflow run segment -b cellpose -i image.tif -o masks.tif
```

---

## B. Developers / power users / HPC

The full control path — unchanged and fully supported.

```bash
# editable install with the light in-core backends + dev tools
python -m venv envs/labflow && envs/labflow/bin/pip install -e ".[light,dev]"
#   (or one command for any machine: python bootstrap.py --extras light,dev)

labflow list                          # every method, grouped by stage
labflow run cluster -b dbscan -i locs.csv -o clusters.csv --param eps=120
labflow install decode                # build/pull a tool's isolated env/image
labflow pipeline --engine snakemake --cores 4
labflow pipeline --engine nextflow  -c conf/mylab.config   # HPC/cloud (see conf/README.md)
pytest labflow/tests driftcorr/tests  # the test suite
```

### Make `labflow` callable from anywhere (no venv activation)

`labflow` is a real entry point, but it lives in the project's env. To run it as plain
`labflow` from any directory in bash:

```bash
bash scripts/install_cli.sh     # writes ~/.local/bin/labflow -> this repo's env
                                # (add ~/.local/bin to PATH if it prints the one-liner)
labflow help                    # now works from anywhere
```

Cross-platform alternative (isolated, on PATH automatically): `pipx install -e .`
(or `pipx install smlm-labflow` once published). Or just activate the env:
`source envs/labflow/Scripts/activate` (Windows) / `source envs/labflow/bin/activate`.

- **Per-tool isolation / GPU / Apptainer:** [docs/environments.md](environments.md)
- **Add a method (one YAML entry):** [docs/methods.md](methods.md)
- **HPC / institutional configs:** [conf/README.md](../conf/README.md)
- **Build + publish the tool images:** [.github/workflows/build-images.yml](../.github/workflows/build-images.yml)
- **Pin images to digests:** `bash scripts/pin_images.sh`

---

## Which gives what

| | conda/mamba | pixi | napari | pip (dev) |
|---|---|---|---|---|
| audience | lab users | lab users / reproducibility | bench scientists (GUI) | developers / HPC |
| locked & cross-platform | partial | **yes (pixi.lock)** | no | via lockfiles |
| GUI | — | — | **yes** | — |
| add a method / full CLI | yes | yes | — | **yes** |

Heavy tools are isolated and fetched on demand via `labflow install` regardless of path,
so no front door pulls a multi-GB dependency you didn't ask for.
