# StarDist segment backend for SMLM LabFlow (CPU; Schmidt et al. 2018).
# Build:  docker build -t smlm-labflow/stardist:latest docker/stardist
# For GPU: swap tensorflow-cpu for the matching tensorflow + a CUDA base, run --gpus all.
FROM python:3.10-slim

# StarDist + csbdeep (normalization) on CPU TensorFlow. Pinned loosely; if you bump
# tensorflow, verify StarDist2D.from_pretrained / predict_instances still match.
RUN pip install --no-cache-dir stardist csbdeep "tensorflow-cpu>=2.10,<2.16" tifffile numpy

COPY run_stardist.py /opt/stardist/run_stardist.py
WORKDIR /work
# No ENTRYPOINT: labflow's runner passes the full command
# (python /opt/stardist/run_stardist.py --in ... --out ...).
