#!/usr/bin/env bash
set -euo pipefail

# Emit a marker before touching mounted storage so application startup failures
# can be distinguished from OCI/sidecar failures that happen before this runs.
echo 'LabFlow startup: entering /start.sh' >&2
trap 'status=$?; printf "LabFlow startup failed at line %s (exit %s)\n" "$LINENO" "$status" >&2' ERR
if [[ $# -eq 0 ]]; then
    set -- sshd
fi

mkdir -p /workspace/{data,outputs,models,backends,envs}
if [[ "${1:-}" != "sshd" ]]; then
    exec "$@"
fi

install -d -m 700 /root/.ssh /workspace/.labflow-ssh
install -d -m 755 /run/sshd
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    key_file=$(mktemp /root/.ssh/authorized_keys.XXXXXX)
    trap 'rm -f "$key_file"' EXIT
    printf '%s\n' "$PUBLIC_KEY" > "$key_file"
    ssh-keygen -l -f "$key_file" >/dev/null
    install -m 600 "$key_file" /root/.ssh/authorized_keys
    rm -f "$key_file"
    trap - EXIT
elif [[ ! -s /root/.ssh/authorized_keys ]]; then
    echo 'No SSH public key configured. Set PUBLIC_KEY in the RunPod template.' >&2
fi

# Generate unique host keys at first boot and retain them with this volume.
host_key=/workspace/.labflow-ssh/ssh_host_ed25519_key
if [[ ! -f "$host_key" ]]; then
    ssh-keygen -q -t ed25519 -N '' -C '' -f "$host_key"
fi
chmod 600 "$host_key"
unset PUBLIC_KEY
echo 'LabFlow ready: labflow doctor | labflow conformance | labflow demo'
exec /usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_labflow
