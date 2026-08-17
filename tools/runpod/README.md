# Tinker server on RunPod

Qwen3.5-9B-Base, 1 trainable LoRA (rank 64), 4096 max tokens, 64 top-k
logprobs, on a 2x RTX PRO 6000 Blackwell (96 GB) pod. Both roles span both
GPUs at TP=2 (trainer `127.0.0.1:8001`, sampler `127.0.0.1:8000`) — see
"Performance" below for why.

Nothing but SSH is exposed publicly — both servers bind loopback, and the
Tinker API neither authenticates nor sandboxes its filesystem writes, so keep
it that way. Access is through an SSH tunnel (`~/.ssh/config` host
`runpod-tinker` forwards local 18001 → trainer, 18000 → sampler), which a
systemd user service holds open permanently:

```bash
systemctl --user status runpod-tinker-tunnel   # should already be running
export TINKER_BASE_URL=http://localhost:18001
python -c "import tinker; print(tinker.ServiceClient(api_key='tml-local').get_server_capabilities())"
```

The unit (`~/.config/systemd/user/runpod-tinker-tunnel.service`) restarts the
tunnel every 10s until it sticks, so it self-heals across drops, pod restarts,
and reboots of this machine (linger is enabled). While the pod is stopped it
just keeps retrying harmlessly.

The `runpod-tinker` ssh entry has no fixed address: a `ProxyCommand`
(`tools/runpod/ssh_proxy.sh`) asks the RunPod API for the pod's current public
IP/port on every connection, and the pod's SSH host keys are persisted in
`/workspace/ssh_host_keys` and restored by `boot.sh`, so stop/start cycles
need no manual ssh-config surgery at all.

## Files

- `env.sh` — every knob (model, ports, LoRA/logprob limits, paths, memory fractions).
- `bootstrap.sh` — idempotent env setup on the pod: uv, two venvs (jax vs vllm
  trees conflict), model download.
- `run_sampler.sh` / `run_trainer.sh` — one role each; tmux window commands.
- `launch.sh` — starts both roles in tmux session `tinker`, waits for health.
- `boot.sh` — pod start command hook: bootstrap + launch on every container start.
- `ssh_proxy.sh` — local ProxyCommand: resolves the pod's current IP/port per connection.

## Operating the pod

```bash
runpodctl get pod                                  # list, find the pod id
runpodctl stop pod <id>                            # ~$0.13/day for disk, GPUs released
runpodctl start pod <id>                           # boot.sh relaunches everything
runpodctl remove pod <id>                          # DESTROYS /workspace state
```

The public IP/port for SSH changes across stop/start, but the `ProxyCommand`
in the ssh config resolves it fresh on every connection — nothing to update.

On the pod: `tmux attach -t tinker` (windows `sampler`, `trainer`);
logs also append to `/workspace/logs/{sampler,trainer}.log`. Everything
persistent (venvs, model, SQLite DB, checkpoints, LoRA adapters, JIT caches)
lives on the volume disk at `/workspace`; the container overlay including all
apt packages is wiped on stop and rebuilt by `boot.sh`.

To redeploy code changes: `rsync` the repo to `/workspace/SkyRL` (exclude
`.venv`, `.git`, logs), then restart the affected role:
`tmux respawn-window -k -t tinker:trainer` (or `:sampler`). To restart the
whole stack, kill the tmux session and re-run `tools/runpod/launch.sh`
(detach it with `setsid nohup ... &` — a bare `nohup ... &` over ssh dies
with the session before tmux exists).

## Performance

Full measurements, topology rationale, and rejected alternatives live in
`runbook.md` (RunPod section + performance notes) — kept there, and only
there, so the numbers can't drift. Headline: co-located TP=2 with the tuned
backend config runs a production step in **146 s vs 232 s** for the original
one-GPU-per-role setup (1.59x), measured 2026-08-16.
