#!/usr/bin/env bash
# Start (or restart) the sampler + trainer under tmux. Idempotent: if the
# trainer already answers healthz, does nothing. Each role runs its own
# run_*.sh as the tmux window command, so a single role restarts with
# `tmux respawn-window -k -t tinker:<role>` without touching the other.
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
source "$DIR/env.sh"

if curl -sf "http://127.0.0.1:${TRAINER_PORT}/api/v1/healthz" >/dev/null 2>&1; then
    echo "trainer already healthy on :${TRAINER_PORT}; nothing to do"
    exit 0
fi

tmux kill-session -t tinker 2>/dev/null || true
tmux new-session -d -s tinker -n sampler "$DIR/run_sampler.sh"
tmux new-window -t tinker -n trainer "$DIR/run_trainer.sh"
# Keep dead panes around so a crash is inspectable instead of a vanished window.
tmux set-option -t tinker remain-on-exit on

echo "launched tmux session 'tinker' (windows: sampler, trainer)"
echo "waiting for readiness..."
deadline=$((SECONDS + 1800))
until curl -sf "http://127.0.0.1:${SAMPLER_PORT}/v1/models" >/dev/null 2>&1; do
    [ $SECONDS -gt $deadline ] && { echo "sampler not ready in 30 min"; exit 1; }
    sleep 5
done
echo "sampler ready"
until curl -sf "http://127.0.0.1:${TRAINER_PORT}/api/v1/healthz" >/dev/null 2>&1; do
    [ $SECONDS -gt $deadline ] && { echo "trainer not ready in 30 min"; exit 1; }
    sleep 5
done
echo "trainer ready -- Tinker API live on 127.0.0.1:${TRAINER_PORT}"
