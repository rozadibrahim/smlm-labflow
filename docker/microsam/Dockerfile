# micro-SAM segment backend for SMLM LabFlow (Segment Anything for microscopy;
# Archit et al. 2024). CPU image; for GPU use a CUDA base + torch-cuda, run --gpus all.
# Build:  docker build -t smlm-labflow/microsam:latest docker/microsam
#
# micro_sam is distributed on conda-forge (pulls SAM + torch), so we use a micromamba
# base. SAM weights download on first run (cached if you pre-fetch into a layer).
FROM mambaorg/micromamba:1.5-jammy

RUN micromamba install -y -n base -c conda-forge micro_sam tifffile numpy && \
    micromamba clean -a -y

COPY run_microsam.py /opt/microsam/run_microsam.py
WORKDIR /work
# The micromamba entrypoint activates the base env, so labflow's
# `python /opt/microsam/run_microsam.py ...` resolves to micro_sam's interpreter.
