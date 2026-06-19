---
title: 'SMLM LabFlow: a registry-driven, reproducible pipeline for single-molecule localization microscopy'
tags:
  - Python
  - single-molecule localization microscopy
  - super-resolution microscopy
  - bioimage analysis
  - reproducibility
  - Snakemake
  - Nextflow
authors:
  - name: Rozad Ibrahim
    orcid: 0000-0000-0000-0000   # TODO: add ORCID
    affiliation: 1
affiliations:
  - name: ESBS, University of Strasbourg, France
    index: 1
date: 19 June 2026
bibliography: paper.bib
---

# Summary

Single-molecule localization microscopy (SMLM) turns raw blinking movies into
super-resolved measurements through a long chain of steps: localization, drift
correction, segmentation, tracking, cluster analysis, spatial statistics, molecular
counting, diffusion analysis, and quality control. Each step has many competing tools,
written in different languages and frameworks with mutually incompatible dependencies
(PyTorch, TensorFlow, Julia, Java/Fiji), and most analyses stitch them together by hand.

**SMLM LabFlow** makes this chain reproducible and composable. Every method — classical
or deep-learning — is declared once in a single registry (`config/methods.yaml`), which
drives one command-line interface and two interchangeable workflow engines (Snakemake
[@snakemake] and Nextflow [@nextflow]). Methods communicate only through a canonical CSV
*file contract*; each runs in its own isolated environment (in-process, virtualenv, conda,
or container) selected per method. Because no two tools share a Python process, backends
on conflicting dependencies coexist in one pipeline without clashes. The same registry
entry exposes a method in the CLI, both pipeline engines, an automatic conformance test,
and provenance records, so adding a tool is a single declarative change.

# Statement of need

SMLM software is highly fragmented and the "glue" is rarely reusable or reproducible.
Comparing two localizers, or running localization with one tool and clustering with
another, typically means bespoke scripts and incompatible conda environments. Wet-lab
groups — the primary users — are further blocked by installation friction and the absence
of a common interface across tools. Existing deep-learning methods (e.g., DECODE
[@decode], Cellpose [@cellpose], StarDist [@stardist]) are excellent individually but ship
as standalone packages with no shared contract.

SMLM LabFlow addresses this with a *narrow-waist* design: the only thing that crosses a
tool boundary is a file, so isolation is automatic and any tool, in any language, docks
the same way. It ships ~50 methods across the localize, drift, segment, track, cluster,
spatial-statistics, counting, analyze, metrics and render stages. Classical, dependency-light
methods run in-process and are validated end-to-end — drift correction (RCC [@rcc], DME
[@dme], AIM [@aim], fiducial), density and tessellation clustering (DBSCAN, SR-Tesseler
[@srtesseler]), spatial statistics (Ripley's K/L, pair- and cross-correlation, coordinate-based
colocalization [@cbc]), qPAINT counting [@qpaint], mean-squared-displacement diffusion
analysis, super-resolution rendering, and the standard resolution/precision metrics
(FRC [@frc], NeNA [@nena]). Heavy deep-learning tools are wired with isolated, version-pinned
environments and prebuilt container images pulled on demand.

Two install paths serve two audiences without compromise: a turnkey, locked path for
wet-lab users (conda-forge, a cross-platform `pixi` lockfile, and a napari [@napari] plugin)
and a full developer/HPC path (pip, the CLI, per-tool containers, and Apptainer/Nextflow
for clusters). A built-in conformance command lets any lab verify, on its own machine,
exactly which methods run.

# Acknowledgements

SMLM LabFlow orchestrates, and is indebted to, the many open-source SMLM and bioimage
tools it wraps, including LiteLoc [@liteloc], DECODE, Cellpose, StarDist, and the
classical methods cited above. We thank the developers of Snakemake, Nextflow, and napari.

# References
