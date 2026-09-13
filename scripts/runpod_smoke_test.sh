#!/usr/bin/env bash
set -euo pipefail
image=${1:?Usage: runpod_smoke_test.sh IMAGE}
scratch=$(mktemp -d)
container=""
volume=$(docker volume create)
cleanup() {
    if [[ -n "$container" ]]; then
        docker logs "$container" || true
        docker rm -f "$container" >/dev/null || true
    fi
    docker volume rm "$volume" >/dev/null || true
    rm -rf "$scratch"
}
trap cleanup EXIT
ssh-keygen -q -t ed25519 -N '' -C 'labflow-ci' -f "$scratch/key"
container=$(docker run -d -p 127.0.0.1::22 \
    -e PUBLIC_KEY="$(cat "$scratch/key.pub")" \
    -v "$volume:/workspace" "$image")
port=$(docker port "$container" 22/tcp | cut -d: -f2)
ssh_args=(-i "$scratch/key" -p "$port" -o BatchMode=yes -o IdentitiesOnly=yes
    -o ConnectTimeout=2 -o StrictHostKeyChecking=accept-new
    -o HostKeyAlias=labflow-ci
    -o "UserKnownHostsFile=$scratch/known_hosts")
ready=false
for attempt in {1..30}; do
    if ssh "${ssh_args[@]}" root@127.0.0.1 true 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { echo 'SSH did not become ready' >&2; exit 1; }
ssh "${ssh_args[@]}" root@127.0.0.1 \
    'labflow --help && python -c "import sys; assert sys.version_info[:2] == (3, 12)" && test -d /workspace/outputs'
ssh "${ssh_args[@]}" root@127.0.0.1 \
    'labflow demo --out /workspace/outputs/smoke && touch /workspace/outputs/persistence-check'
ssh "${ssh_args[@]}" root@127.0.0.1 \
    'labflow pipeline -n --printshellcmds' > "$scratch/workflow-plan"
grep -F '/opt/conda/envs/smlm-labflow/bin/python run_pipeline.py infer' "$scratch/workflow-plan"
grep -F '/workspace/outputs/snakemake_run' "$scratch/workflow-plan"
docker restart "$container" >/dev/null
# Docker may reassign an ephemeral published port when the container restarts.
port=$(docker port "$container" 22/tcp | cut -d: -f2)
ssh_args[3]="$port"
# Same known_hosts file proves that restarting preserves the host identity.
ready=false
for attempt in {1..30}; do
    if ssh "${ssh_args[@]}" root@127.0.0.1 \
        'test -f /workspace/outputs/persistence-check' 2>"$scratch/restart-error"; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { cat "$scratch/restart-error" >&2; echo 'Restart/persistence check failed' >&2; exit 1; }
# Simulate a platform-provided shell command, independently of Docker's CMD.
# The same volume and known_hosts must still work after replacing the container.
docker rm -f "$container" >/dev/null
container=$(docker run -d -p 127.0.0.1::22 \
    -e PUBLIC_KEY="$(cat "$scratch/key.pub")" \
    -v "$volume:/workspace" "$image" /bin/bash -c 'exec /start.sh')
port=$(docker port "$container" 22/tcp | cut -d: -f2)
ssh_args[3]="$port"
ready=false
for attempt in {1..30}; do
    if ssh "${ssh_args[@]}" root@127.0.0.1 \
        'test -f /workspace/outputs/persistence-check && command -v labflow' 2>"$scratch/override-error"; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { cat "$scratch/override-error" >&2; echo 'Injected startup command failed' >&2; exit 1; }
docker logs "$container" 2>&1 | grep -F 'LabFlow startup: entering /start.sh'
docker run --rm "$image" labflow-liteloc --help
