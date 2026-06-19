# DeepSPT diffusional-fingerprinting backend for SMLM LabFlow (Jacobsen et al. 2024). GPU.
# Build:  docker build -t smlm-labflow/deepspt:latest docker/deepspt
# Runs with --gpus all automatically (method has gpu: true); falls back to CPU.
#
# Placeholder deps (torch base + sklearn for the classifier). DeepSPT's exact package +
# weights are the binding point (run_deepspt.py); pin them here when wiring the model.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir pandas numpy scikit-learn

COPY run_deepspt.py /opt/deepspt/run_deepspt.py
WORKDIR /work
# No ENTRYPOINT: labflow's runner passes the full command.
