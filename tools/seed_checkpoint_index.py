"""Re-register orphaned Tinker checkpoints in the session database.

Needed for checkpoints written while the server's `--database-url` pointed at a
container-local SQLite file. Their Orbax directories survive on the state Volume,
but the rows `validate_checkpoint` looks up (api.py:1175) were destroyed with the
container, so `load_weights` 404s with "Model not found" and the run cannot be
resumed despite the weights being intact.

This rebuilds the minimum set of rows -- session, model, checkpoint -- to make
those checkpoints addressable again. Optimizer state is not reconstructed because
it was never lost: it lives inside the Orbax checkpoint alongside the LoRA
weights (jax.py:1127-1131). Only the index was missing.

Runs inside the container's repo venv, where sqlmodel and skyrl are installed.

    python tools/seed_checkpoint_index.py \
        --db-path /state/tinker.db --ckpt-dir /state/checkpoints \
        --base-model /models/Qwen/Qwen3.5-9B-Base \
        --model-id model_705230e0 --checkpoint-ids atk_000128
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from skyrl.tinker import types
from skyrl.tinker.db_models import CheckpointDB, CheckpointStatus, ModelDB, SessionDB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint-ids", required=True, help="comma-separated")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_dir) / args.model_id
    if not ckpt_root.is_dir():
        raise SystemExit(f"no checkpoint directory {ckpt_root}; nothing to recover")

    engine = create_engine(f"sqlite:///{args.db_path}")
    SQLModel.metadata.create_all(engine)  # no-op if the server already made them

    session_id = f"recovered_{args.model_id}"
    now = datetime.now(timezone.utc)
    added: list[str] = []

    with Session(engine) as db:
        if db.get(SessionDB, session_id) is None:
            db.add(SessionDB(session_id=session_id, sdk_version="recovered", status="active"))

        existing = db.get(ModelDB, args.model_id)
        if existing is not None:
            # base_model is echoed back to the client by /weights_info and used
            # there to build a tokenizer, so it must be the HF repo id the client
            # originally sent -- not the server-side container path. Correct it in
            # place; a stale value fails client-side with HFValidationError.
            if existing.base_model != args.base_model:
                print(f"  fix  base_model {existing.base_model!r} -> {args.base_model!r}")
                existing.base_model = args.base_model
                db.add(existing)
        else:
            db.add(
                ModelDB(
                    model_id=args.model_id,
                    base_model=args.base_model,
                    # alpha is hardcoded to 32.0 server-side (api.py:799). The rank
                    # must match the checkpoint or _insert_checkpoint_data raises
                    # "Rank mismatch" on load (jax.py:1143).
                    lora_config=types.LoraConfig(rank=args.rank, alpha=32.0, seed=args.seed).model_dump(),
                    status="created",
                    request_id=0,
                    session_id=session_id,
                )
            )

        for checkpoint_id in [c.strip() for c in args.checkpoint_ids.split(",") if c.strip()]:
            # The Orbax checkpoint is a *directory* named <id>.tar.gz, not an archive.
            if not (ckpt_root / f"{checkpoint_id}.tar.gz").exists():
                print(f"  SKIP {checkpoint_id}: not present on the Volume")
                continue
            key = (args.model_id, checkpoint_id, types.CheckpointType.TRAINING)
            if db.get(CheckpointDB, key) is not None:
                print(f"  ok   {checkpoint_id}: row already present")
                continue
            db.add(
                CheckpointDB(
                    model_id=args.model_id,
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=types.CheckpointType.TRAINING,
                    status=CheckpointStatus.COMPLETED,
                    created_at=now,
                    completed_at=now,
                )
            )
            added.append(checkpoint_id)
        db.commit()

        rows = db.exec(select(CheckpointDB).where(CheckpointDB.model_id == args.model_id)).all()

    print(f"added={added}")
    print(f"indexed={sorted((r.checkpoint_id, r.checkpoint_type.value, r.status.value) for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
