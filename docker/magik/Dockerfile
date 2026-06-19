# MAGIK GNN tracking backend for SMLM LabFlow (DeepTrackAI, Pineda et al. 2023). GPU.
# Build:  docker build -t smlm-labflow/magik:latest docker/magik
# Runs with --gpus all automatically (method has gpu: true); falls back to CPU.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# MAGIK ships in DeepTrack (deeptrack). pandas/numpy for the file contract IO.
RUN pip install --no-cache-dir deeptrack pandas numpy tifffile

COPY run_magik.py /opt/magik/run_magik.py
WORKDIR /work
# No ENTRYPOINT: labflow's runner passes the full command.
