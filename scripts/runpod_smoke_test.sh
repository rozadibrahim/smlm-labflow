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
docker restart "$container" >/dev/null
# Same known_hosts file proves that restarting preserves the host identity.
ready=false
for attempt in {1..30}; do
    if ssh "${ssh_args[@]}" root@127.0.0.1 \
        'test -f /workspace/outputs/persistence-check' 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { echo 'Restart/persistence check failed' >&2; exit 1; }
docker run --rm "$image" labflow-liteloc --help
