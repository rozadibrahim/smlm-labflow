# Recording an experimental run

A useful run record connects the scientific question to the files and commands that produced the result. Keep it next to the corresponding configuration and small, shareable outputs.

## Context

State the question the run addresses, the sample or structure being imaged, and what part of the work was performed during the M1 internship. Describe the tasks performed in LabFlow separately from the localization method provided by LiteLoc. Include one implementation decision or debugging observation from the actual work if it helps explain the result.

## Input and environment

Record the dataset source or accession, permission to redistribute it, selected files/frames, and any preprocessing. For public inputs, provide a stable download link and checksum. Include the LabFlow commit, LiteLoc commit, environment, GPU, acquisition settings, and model/calibration provenance.

## Reproduction

Commit the scientific profile without machine-specific paths and show the exact commands. Explain how the calibration and model were selected, including whether they were reused from the registry. Keep raw data and model weights outside Git when they are large or cannot be shared.

## Result

Include a small localization table, a representative figure, and the relevant QC/report extracts. Give coordinate units, frame indexing, thresholds, and whether uncertainty fields were measured or filled with defaults. Record the actual runtime and hardware if discussing speed.

An accuracy comparison needs reference localizations or ground truth, the matching procedure/tolerance, and the same evaluation conditions for each method. Without a reference, report the observations and QC available without calling them accuracy measurements.

## Interpretation

Explain what the run supports, what did not work, and what you would check next. A synthetic software example should remain labeled synthetic; it cannot stand in for an experimental internship result.
