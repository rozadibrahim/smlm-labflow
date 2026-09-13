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
# Model the RunPod volume behavior: chmod succeeds but leaves workspace files
# at 0666. Seed the host key written by earlier images and leave it untouched.
cat > "$scratch/chmod" <<'SH'
#!/bin/sh
for arg in "$@"; do
    case "$arg" in /workspace/*) exit 0 ;; esac
done
exec /bin/chmod "$@"
SH
chmod 755 "$scratch/chmod"
docker run --rm -v "$volume:/workspace" --entrypoint /bin/bash "$image" -c \
    'mkdir -p /workspace/.labflow-ssh; ssh-keygen -q -t ed25519 -N "" -C "" -f /workspace/.labflow-ssh/ssh_host_ed25519_key; chmod 666 /workspace/.labflow-ssh/ssh_host_ed25519_key'
volume_behavior=(-v "$scratch/chmod:/usr/local/bin/chmod:ro")
if [[ -n "${2:-}" ]]; then
    container=$(docker run -d -e PUBLIC_KEY="$(cat "$scratch/key.pub")" \
        -v "$volume:/workspace" "${volume_behavior[@]}" "$2")
    for attempt in {1..15}; do
        [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == false ]] && break
        sleep 1
    done
    [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == false ]]
    docker logs "$container" > "$scratch/baseline.log" 2>&1
    grep -F 'bad permissions' "$scratch/baseline.log"
    grep -F 'no hostkeys available' "$scratch/baseline.log"
    docker rm -f "$container" >/dev/null
    container=""
    echo 'Confirmed: previous image cannot start SSH on a volume that ignores chmod.'
fi
container=$(docker run -d -p 127.0.0.1::22 \
    -e PUBLIC_KEY="$(cat "$scratch/key.pub")" \
    -v "$volume:/workspace" "${volume_behavior[@]}" "$image")
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
docker exec "$container" /bin/bash -c \
    'test "$(stat -c %a /workspace/.labflow-ssh/ssh_host_ed25519_key)" = 666 && test "$(stat -c %a /etc/ssh/labflow/ssh_host_ed25519_key)" = 600'
docker exec "$container" ssh-keygen -lf /etc/ssh/labflow/ssh_host_ed25519_key > "$scratch/first-host"
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
# Workspace data survives replacement, but the new container has a new local
# server identity. Use a fresh alias in the test trust store, keeping checks on.
docker rm -f "$container" >/dev/null
container=$(docker run -d -p 127.0.0.1::22 \
    -e PUBLIC_KEY="$(cat "$scratch/key.pub")" \
    -v "$volume:/workspace" "${volume_behavior[@]}" "$image" /bin/bash -c 'exec /start.sh')
port=$(docker port "$container" 22/tcp | cut -d: -f2)
ssh_args[3]="$port"
for index in "${!ssh_args[@]}"; do
    if [[ "${ssh_args[$index]}" == HostKeyAlias=labflow-ci ]]; then
        ssh_args[$index]=HostKeyAlias=labflow-ci-replacement
    fi
done
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
docker exec "$container" ssh-keygen -lf /etc/ssh/labflow/ssh_host_ed25519_key > "$scratch/replacement-host"
if cmp -s "$scratch/first-host" "$scratch/replacement-host"; then
    echo 'A replacement container must generate its own local SSH host key' >&2
    exit 1
fi
docker logs "$container" 2>&1 | grep -F 'LabFlow startup: entering /start.sh'
docker run --rm "$image" labflow-liteloc --help
