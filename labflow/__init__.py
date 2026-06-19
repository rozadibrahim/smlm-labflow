"""
labflow: the harmonic spine for SMLM LabFlow.

One method registry (config/methods.yaml) drives one CLI (`labflow`) and one
Snakemake pipeline. Every method -- localizer, drift corrector, tracker,
analyser, in any language or environment -- docks the same way over a canonical
CSV file contract. Adding a method is a single registry entry that appears in
both the CLI and the pipeline.
"""

from .registry import load_registry, methods, resolve, stages
from .runner import run_method

__all__ = ["load_registry", "methods", "resolve", "stages", "run_method"]
