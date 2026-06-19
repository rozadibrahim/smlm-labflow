# Omnipose segment backend for SMLM LabFlow (CPU; Cutler et al. 2022).
# Build:  docker build -t smlm-labflow/omnipose:latest docker/omnipose
# For GPU: use a CUDA base + torch-cuda and run with --gpus all.
FROM python:3.10-slim

# Omnipose pulls its patched cellpose fork (cellpose_omni) + torch (CPU here).
RUN pip install --no-cache-dir omnipose tifffile numpy

COPY run_omnipose.py /opt/omnipose/run_omnipose.py
WORKDIR /work
# No ENTRYPOINT: labflow's runner passes the full command.
