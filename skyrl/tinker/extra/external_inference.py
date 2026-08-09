import asyncio
import errno
import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from cloudpathlib import AnyPath
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import convert_vllm_prompt_logprobs
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack

if TYPE_CHECKING:
    from skyrl.tinker.api import SampleRequest


def _extract_checkpoint_sync(checkpoint_path: AnyPath, target_dir: Path) -> None:
    """Extract a LoRA checkpoint to disk for vLLM to load.

    The checkpoint is extracted onto ``target_dir``'s own filesystem (via
    ``scratch_dir``), so publishing it is a single atomic ``rename``. This is
    what prevents the load-time 404: a concurrent request racing to load the
    same freshly-trained checkpoint either sees no ``target_dir`` or sees the
    complete one, never a half-written directory. Extracting into the system
    temp dir instead makes the rename a cross-device move, which either raises
    EXDEV or degrades to a non-atomic copy that briefly exposes a partial
    directory.

    This is a blocking operation (filesystem/network I/O) and should be called
    via asyncio.to_thread() to avoid blocking the event loop.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Extract the checkpoint if it doesn't already exist
    if not target_dir.exists():
        try:
            # Extract onto target_dir's filesystem so the rename below is atomic.
            with download_and_unpack(checkpoint_path, scratch_dir=target_dir.parent) as extracted_path:
                extracted_path.rename(target_dir)
        except OSError as e:
            # This could happen if two processes try to download the file.
            # In that case the other process won the race and created target_dir.
            if not target_dir.exists() or e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.base_url = f"{engine_config.external_inference_url}/v1"
        self.api_key = engine_config.external_inference_api_key
        self.config = engine_config
        self.db_engine = db_engine
        # RL refreshes the sampler adapter every optimizer step, each under a new
        # checkpoint_id, and vLLM's filesystem resolver loads them on demand but never
        # unloads -- so adapters pile up (one per step) until the engine wedges (GPU
        # pinned, no completions). Track the live adapter per model and unload the prior
        # one when it rolls over, so vLLM only ever holds the current adapter.
        self._live_lora: dict[str, tuple[int, str]] = {}
        self._lora_seq = itertools.count()
        self._lora_lock = asyncio.Lock()

    async def _retire_previous_adapter(
        self, model_id: str, model_name: str, seq: int, http_client: httpx.AsyncClient
    ) -> None:
        """Unload the model's previous adapter now that ``model_name`` has served a request.

        ``seq`` is the order this request *started* in, not the order it finished. Samples
        for consecutive checkpoints overlap, so ranking by completion lets a straggler from
        the older checkpoint declare itself live and unload the newer adapter that requests
        are still arriving for -- while the genuinely stale one leaks, which is the opposite
        of the point. Start order is the closest signal available here: checkpoint ids are
        caller-supplied strings with no guaranteed ordering, and a sample cannot start
        before its adapter was published.

        Best-effort: a failed unload just reverts to the old accumulate-and-wedge
        behaviour, it never breaks sampling. The lock serialises only the bookkeeping,
        so exactly one sample per rollover issues the unload; the rest do not wait on it.
        """
        async with self._lora_lock:
            live = self._live_lora.get(model_id)
            if live is not None and (live[1] == model_name or live[0] > seq):
                # Already the live adapter, or an older request finishing late -- either way
                # there is nothing this call should unload.
                return
            self._live_lora[model_id] = (seq, model_name)
        if live is None:
            return
        previous = live[1]
        try:
            response = await http_client.post("/unload_lora_adapter", json={"lora_name": previous})
            response.raise_for_status()
        except httpx.HTTPError:
            # Non-fatal: worst case the old adapter lingers, which is just the prior
            # (working-but-leaky) behaviour.
            logger.warning(f"Failed to unload stale LoRA adapter {previous}; it will linger on the engine")

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Background task to call external engine and store result in database."""
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 minutes for inference, 10s for connect
            ) as http_client:
                result = await self._forward_to_engine(
                    sample_req, model_id, checkpoint_id, http_client, base_model=base_model
                )
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_to_engine(
        self,
        request: "SampleRequest",
        model_id: str,
        checkpoint_id: str,
        http_client: httpx.AsyncClient,
        *,
        base_model: str | None = None,
    ) -> types.SampleOutput:
        """Forward request to vLLM with dynamic LoRA loading.

        Extracts the checkpoint to the configured external_inference_lora_base and references it by a model name
        that vLLM can dynamically load via the lora_filesystem_resolver plugin.

        For base model sampling (no LoRA), the request is sent directly using the base model name.
        """
        model_input = request.prompt.to_types()
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        # Stamped before the call so adapter retirement can rank requests by start order
        # rather than completion order; see _retire_previous_adapter.
        lora_seq = next(self._lora_seq)

        if base_model:
            # Base model sampling: use the model name directly, no LoRA checkpoint needed
            model_name = base_model
        else:
            # LoRA sampling: reference the adapter by name for dynamic loading.
            target_dir = self.config.sampler_adapter_dir(model_id, checkpoint_id)
            model_name = target_dir.name

            # The adapter may already be published here as a plain directory (see
            # EngineConfig.publishes_sampler_adapter_in_place), in which case no extraction is
            # needed. Otherwise extract the tar.gz -- also the path for pre-existing sampler
            # checkpoints written before in-place publishing.
            if not target_dir.exists():
                checkpoint_path = self.config.sampler_archive_path(model_id, checkpoint_id)
                await asyncio.to_thread(_extract_checkpoint_sync, checkpoint_path, target_dir)

        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": request.num_samples,
            "seed": request.sampling_params.seed,
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "top_p": request.sampling_params.top_p,
            "top_k": request.sampling_params.top_k,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
        }
        # vLLM's `prompt_logprobs` is an int: 0 returns just the prompt tokens'
        # own logprobs, k>0 also returns the top-k per position.
        topk_prompt_logprobs = getattr(request, "topk_prompt_logprobs", 0) or 0
        want_prompt_logprobs = bool(request.prompt_logprobs) or topk_prompt_logprobs > 0
        if want_prompt_logprobs:
            payload["prompt_logprobs"] = topk_prompt_logprobs

        # Pass X-Session-ID for deterministic routing
        headers = {}
        session_id = types.make_routing_session_id(request.sampling_session_id, request.seq_id)
        if session_id is not None:
            headers["X-Session-ID"] = session_id

        response = await http_client.post("/completions", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # The request succeeded, so this adapter is definitely resident; retire the one it
        # replaced. Done after the call rather than before so a failed sample never unloads
        # a working adapter.
        if not base_model:
            await self._retire_previous_adapter(model_id, model_name, lora_seq, http_client)

        prompt_logprobs = None
        topk = None
        if want_prompt_logprobs:
            # All `n` choices share one prompt, so vLLM repeats the same prompt
            # logprobs on each choice; read them off the first.
            raw = result["choices"][0].get("prompt_logprobs") if result["choices"] else None
            if raw is None:
                logger.warning("Requested prompt logprobs but vLLM /completions returned none")
            prompt_logprobs, topk = convert_vllm_prompt_logprobs(prompt_tokens, raw, topk=topk_prompt_logprobs)

        sequences = []
        for choice in result["choices"]:
            lp = choice["logprobs"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=choice["token_ids"],
                    logprobs=lp["token_logprobs"],
                    stop_reason=choice["finish_reason"],
                )
            )

        return types.SampleOutput(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk,
        )
