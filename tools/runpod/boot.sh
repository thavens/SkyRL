#!/usr/bin/env bash
# Pod entry point: the docker start command runs this on every container start
# (first boot and every stop/start cycle), so it must rebuild whatever the
# container wipe destroyed and then bring the servers up.
set -euo pipefail
cd "$(dirname "$0")"

# Keep SSH host keys stable across container rebuilds: the overlay wipe on
# stop/start regenerates /etc/ssh keys, which makes every client refuse the
# "changed" host. First boot saves the generated keys; later boots restore
# them and HUP the sshd master so it re-execs with the restored keys (safe
# here: this runs from the docker start command before any session exists).
KEYDIR=/workspace/ssh_host_keys
if [ -f "$KEYDIR/ssh_host_ed25519_key" ]; then
    cp "$KEYDIR"/ssh_host_* /etc/ssh/
    chmod 600 /etc/ssh/ssh_host_*_key
    pkill -HUP -x sshd 2>/dev/null || true
else
    mkdir -p "$KEYDIR"
    cp /etc/ssh/ssh_host_* "$KEYDIR"/
fi

./bootstrap.sh
./launch.sh
