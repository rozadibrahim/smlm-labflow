# DeepTRACE track-analysis backend for SMLM LabFlow (DL, 2026). GPU.
# Build:  docker build -t smlm-labflow/deeptrace:latest docker/deeptrace
# Runs with --gpus all automatically (method has gpu: true); falls back to CPU.
#
# Placeholder deps: DeepTRACE's exact package is the binding point (run_deeptrace.py).
# Pin them here when wiring the model.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir pandas numpy

COPY run_deeptrace.py /opt/deeptrace/run_deeptrace.py
WORKDIR /work
# No ENTRYPOINT: labflow's runner passes the full command.
