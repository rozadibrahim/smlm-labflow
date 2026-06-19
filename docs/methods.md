# Methods, stages, and the one way to add a tool

SMLM LabFlow is organized as a chain of **stages**, each consuming the previous
stage's CSV (the *file contract*):

```
calibrate → train → localize → drift → track → analyze → report
```

Every method that implements a stage is declared once in **`config/methods.yaml`**
(the single source of truth). Both the CLI (`labflow`) and the Snakemake pipeline
read that registry, so **adding a method is one YAML entry that appears in both.**

## The file contract (schema.py)

| Stage | Input | Output |
|---|---|---|
| localize | raw movies | `canonical_localizations.csv` (`CANONICAL_COLUMNS`) |
| drift | localizations | `drift_corrected_localizations.csv` (+ trajectory, json) |
| track | localizations | `tracks.csv` (`TRACK_COLUMNS`) |
| analyze | tracks | `track_analysis.csv` (`ANALYSIS_COLUMNS`) |
| report | all of the above | `run_report.html/md` |

A method only ever sees *CSV in → CSV out*. How it runs (language, deps, GPU) is
its own business, kept isolated by its `runtime`.

## Runtimes (how a method is isolated)

| `runtime` | How it runs | Use for |
|---|---|---|
| `python` | imported callable, in-process | light native methods (drift via `driftcorr`) |
| `local` | subprocess on PATH | tools already installed (e.g. Julia AIM) |
| `venv` | subprocess with a per-method virtualenv on PATH | Python tools with conflicting pins |
| `conda` | `conda run -n <env> …` | Python/R tools + tricky system/CUDA deps |
| `docker` | `docker run … <image> …` | full reproducibility, or any awkward runtime |
| `external` | command run directly on the given paths | stage that isn't CSV-in/out (localization) |

No daemon is required unless you choose `docker`. This is how MAGIK (TensorFlow),
DeepTRACE, LiteLoc (torch) and Julia coexist without dependency conflicts.

## Use it

```bash
pip install -e .                                   # one-time; gives the `labflow` CLI

labflow list                                       # every method, grouped by stage
labflow list --stage drift

labflow run drift --method aim_julia -i locs.csv -o out/corrected.csv --param track_interval=500
labflow run track --method magik     -i corrected.csv -o tracks.csv

labflow pipeline --cores 4                          # the whole DAG (snakemake, default)
labflow pipeline -n                                 # dry-run / print the DAG
snakemake -s workflow/Snakefile --config drift_method=aim_julia track=true --cores 4
```

### Two engines, one registry

The pipeline has **two interchangeable emitters** driven by the same
`config/methods.yaml`: `workflow/Snakefile` (Python-native, the default) and
`workflow/main.nf` (Nextflow, for cluster/cloud executors). Both shell the same
`labflow run <stage>` commands and write the same output tree — pick per deployment:

```bash
labflow pipeline --engine snakemake --cores 4                 # laptop / single pod (no JVM)
labflow pipeline --engine nextflow  --profile slurm           # HPC cluster   (needs Java + nextflow)
labflow pipeline --engine nextflow  --profile awsbatch        # cloud
nextflow run workflow/main.nf -c workflow/nextflow.config --drift_method aim_julia
```

Nextflow is optional (it needs Java 11+ and the `nextflow` binary; it is **not** a
pip dependency). If it isn't installed, `--engine nextflow` prints how to get it.
Tool isolation, the file contract, and provenance are handled by `labflow run`
regardless of engine, so the engine only chooses *where* jobs run.

## Add a method (the whole recipe)

1. Add one entry to `config/methods.yaml`:

   ```yaml
   my_tracker:
     stage: track
     runtime: conda            # or venv / docker / local / python
     env: smlm-mytracker       # conda env name (or venv path)
     command: ["python", "/path/run.py", "--in", "{input}", "--out", "{output}",
               "--params", "{params_json}"]
     input: localizations.csv
     output: tracks.csv
     params: {radius: 1.5}
     description: "My tracker."
   ```

   Tokens in `command`: `{input}` `{output}` `{repo}` `{params_json}` `{p.<key>}`.

2. That's it. It now appears in `labflow list`, runs via
   `labflow run track --method my_tracker -i … -o …`, and is selectable in the
   pipeline (`--config track_method=my_tracker`). No other code changes.

For an **in-process Python** method (light, dependency-compatible), instead set
`runtime: python`, `entry: "labflow.stages.your_stage:run"`, and implement
`run(*, input_csv, output_csv, params)`. **Start from the annotated template
[`labflow/stages/_template.py`](../labflow/stages/_template.py)** — copy it and
write the science. Plumb files through the spine's IO so they stay on the
canonical contract instead of re-implementing CSV handling:

```python
from labflow.io import read_localizations, read_table, write_table

def run(*, input_csv, output_csv, params):
    locs = read_localizations(input_csv)   # alias-aware (accepts ThunderSTORM/Picasso headers)
    ...                                     # the science
    return write_table(result, output_csv) # creates parent dirs, returns the path
```

For a **family** that shares one adapter (dbscan/optics/hdbscan, the drift
backends, the spatial-stats metrics), give each registry entry the same `entry:`
plus a `bind:` selecting the variant, and branch on it in `run()` — see
`labflow/stages/cluster.py`, `spatial_stats.py`, and `drift.py` (which reuses
`driftcorr`).

After editing the registry, run **`labflow doctor`** — it lints every entry
(missing `entry`/`command`/`env`/`image`, unknown stage or runtime) so a malformed
method fails fast with a clear message instead of crashing mid-run.

## Worked examples already in the registry

- `rcc`, `aim_julia`, `dme`, `none` — drift, `runtime: python` via `driftcorr`
  (`aim_julia` itself shells to the Julia port).
- `liteloc` — localize, `runtime: external`, wraps `run_pipeline.py infer`.
- `magik` (track), `deeptrace` (analyze) — `runtime: docker`, stub images; build
  the image + entrypoint and they work unchanged.
- `dbscan` / `optics` / `hdbscan` (cluster) — `runtime: python`, in-process sklearn.
- `miro` (cluster) — DL (DeepTrackAI, Nat. Commun. 2025): an rGNN transform + DBSCAN,
  `runtime: venv`; needs its own env + a trained model (binding point in
  `labflow/stages/miro_run.py`).

## Review a run in napari (the GUI exception)

The pipeline runs headless, but you open a run in the **full napari GUI** through
labflow for interactive inspection:

```bash
pip install "napari[all]"
labflow review outputs/snakemake_run/infer/results --with napari
```

It loads localizations / drift-corrected localizations / clusters as napari point
layers (clusters coloured by `cluster_id`). Read-only review — napari stays the
fully interactive viewer; labflow just launches it with your run loaded.

### `--gui`: open a backend's own GUI for hands-on use

Any backend can be opened in *its own* GUI (full feature set) instead of running
headless, via `--gui`:

```bash
labflow run segment -b cellpose --gui -i image.tif    # opens the Cellpose GUI
labflow run cluster -b dbscan   --gui -i locs.csv      # no native GUI -> opens napari
```

A method declares how in its `gui:` field (`napari`, or a launch command like
`["cellpose"]` / `["fiji","{input}"]`). If a backend has no `gui:`, `--gui` falls
back to napari, so every backend still gives the user a hands-on view. With
`--gui`, `-o` is optional (you save from the tool's GUI). `--gui` works even for
`status: planned` methods — you get the tool's full GUI once its env is installed.

## Adding a deep-learning method's environment (e.g. MIRO)

DL tools get their own isolated env so they never clash with the core:

```bash
# example: the miro env referenced by config/methods.yaml (runtime: venv, env: envs/miro)
git clone https://github.com/DeepTrackAI/MIRO
python -m venv envs/miro
envs/miro/Scripts/pip install -r MIRO/requirements.txt   # installs deeplay/torch
```

Then `labflow run cluster --method miro -i locs.csv -o clusters.csv --param model=<trained_model>`
runs MIRO inside that env over the file contract — the core install stays untouched.
