# TrackMate tracking backend for SMLM LabFlow (headless Fiji; Tinevez 2017 / Ershov 2022).
# Build:  docker build -t smlm-labflow/trackmate:latest docker/trackmate
# CPU; bundles Fiji (ImageJ) so the lab needs no local Fiji install.
FROM eclipse-temurin:11-jre

# Fetch Fiji (Linux). Verify the URL if the download layout changes.
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && \
    curl -fL -o /tmp/fiji.zip https://downloads.imagej.net/fiji/latest/fiji-linux64.zip && \
    unzip -q /tmp/fiji.zip -d /opt && rm /tmp/fiji.zip && \
    apt-get purge -y curl unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/Fiji.app:$PATH
COPY run_trackmate.py /opt/trackmate/run_trackmate.py
WORKDIR /work
# labflow runs:  ImageJ-linux64 --headless --console --run /opt/trackmate/run_trackmate.py "input='...',output='...'"
