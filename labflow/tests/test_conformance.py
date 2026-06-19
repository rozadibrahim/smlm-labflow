"""Conformance gate: the dependency-self-contained core tools really run end-to-end.

These tools live in the light core (or need only numpy/pandas); whenever they are
installed they MUST pass the synthetic smoke test. Heavy/external tools are allowed
to SKIP (not installed) without failing the suite.

Run: python -m pytest labflow/tests/test_conformance.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from labflow.conformance import run_conformance

# In-core / light tools: if present on this machine, conformance must be PASS.
MUST_PASS = {
    "none", "rcc", "fiducial",                       # drift (no extra deps)
    "dbscan", "optics", "hdbscan", "locan", "srtesseler",   # cluster
    "ripley", "paircorrelation", "nnd", "voronoi",   # spatial_stats
    "gfunction", "cbc", "crosscorrelation",          # spatial_stats (G-function + 2-colour)
    "trackpy",                                        # track
    "qpaint", "blink",                                # counting
    "msd",                                            # analyze (MSD diffusion)
    "render",                                          # render (SR image)
    "frc", "nena",                                    # metrics (resolution + precision)
    "randomforest",                                   # qc_audit
    "_selftest_local",                               # local subprocess mechanism
}


def test_core_tools_pass_conformance():
    results = {r.name: r for r in run_conformance()}
    assert results, "conformance produced no results"
    # The harness must actually have exercised something end-to-end.
    assert any(r.status == "PASS" for r in results.values()), \
        "no tool passed - is the core env installed?"
    failures = [f"{n}: {r.detail}" for n, r in results.items()
                if n in MUST_PASS and r.status == "FAIL"]
    assert not failures, "core conformance failures:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    test_core_tools_pass_conformance()
    print("conformance gate passed")
