# Institutional configs (run the pipeline on *your* cluster)

The Nextflow pipeline ([`workflow/main.nf`](../workflow/main.nf)) is portable: each site
adds a small config describing its scheduler, resources, and container engine, and the
pipeline runs there unchanged — the same pattern [nf-core](https://nf-co.re) uses to run
across institutions.

## Use
1. Copy the template and edit it for your cluster:
   ```bash
   cp conf/institutional.config.template conf/mylab.config
   # edit: executor (slurm/sge/...), queue, GPU options, apptainer cacheDir, data roots
   ```
2. Run, layering your config on top:
   ```bash
   labflow pipeline --engine nextflow -c conf/mylab.config
   # or: nextflow run workflow/main.nf -c workflow/nextflow.config -c conf/mylab.config
   ```

The built-in profiles (`-profile slurm | awsbatch | apptainer`) in
[`workflow/nextflow.config`](../workflow/nextflow.config) are starting points; your
`conf/<lab>.config` overrides them. On HPC use **Apptainer/Singularity** (no root) — the
GHCR tool images are pulled as `.sif` automatically.

## Contribute it back
If your config is reusable (a shared cluster), open a PR adding `conf/<institution>.config`
so other groups on the same system get it for free.
