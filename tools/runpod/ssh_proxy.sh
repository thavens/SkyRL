#!/usr/bin/env bash
# ProxyCommand for the local ~/.ssh/config entry: resolves the pod's *current*
# public SSH endpoint from the RunPod API at connect time, so the entry never
# goes stale when a stop/start cycle reassigns the IP and port.
#
#   ProxyCommand /path/to/ssh_proxy.sh <pod-id>
set -euo pipefail
POD_ID="${1:?usage: ssh_proxy.sh <pod-id>}"
APIKEY=$(grep -oP "(?<=apikey = ').*(?=')" ~/.runpod/config.toml)
read -r IP PORT < <(
    curl -s --max-time 15 "https://api.runpod.io/graphql?api_key=${APIKEY}" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { runtime { ports { ip publicPort privatePort type } } } }\"}" \
    | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22 and .type == "tcp") | "\(.ip) \(.publicPort)"'
)
[ -n "${IP:-}" ] || { echo "ssh_proxy: pod ${POD_ID} has no public SSH port (stopped?)" >&2; exit 1; }
exec nc "$IP" "$PORT"
