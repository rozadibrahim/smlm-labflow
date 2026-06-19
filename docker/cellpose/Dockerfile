# Cellpose segment backend for SMLM LabFlow (CPU).
# Build:  docker build -t smlm-labflow/cellpose:latest docker/cellpose
# For GPU: swap the base for an nvidia/cuda image + torch-cuda, and run with --gpus all.
FROM python:3.11-slim

# Pinned to the Cellpose 3.x API that run_cellpose.py targets (eval -> masks,flows,styles).
RUN pip install --no-cache-dir "cellpose>=3,<4" tifffile numpy
# slim base lacks `packaging`, which fastremap (a cellpose dep) imports at runtime.
RUN pip install --no-cache-dir packaging

COPY run_cellpose.py /opt/cellpose/run_cellpose.py
WORKDIR /work
# No ENTRYPOINT: labflow's runner passes the full command
# (python /opt/cellpose/run_cellpose.py --in ... --out ...).
