import asyncio
import errno
from datetime import datetime, timezone
from pathlib import Path

import httpx
from cloudpathlib import AnyPath
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack


def _extract_checkpoint_sync(checkpoint_path: AnyPath, target_dir: Path) -> None:
    """Extract a LoRA checkpoint to disk for vLLM to load.

    The checkpoint is extracted onto ``target_dir``'s own filesystem (via
    ``scratch_dir``), so publishing it is a single atomic ``rename``. This is
    what prevents the load-time 404: a concurrent request racing to load the
    same freshly-trained checkpoint either sees no ``target_dir`` or sees the
    complete one, never a half-written directory. (The previous version
    extracted into the system temp dir and, on the resulting cross-device
    EXDEV, fell back to a non-atomic ``shutil.move`` straight into
    ``target_dir`` that briefly exposed a partial directory -> 404.)

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

    # Transient failures worth retrying. Other HTTP errors (4xx) are
    # deterministic (bad request, adapter registry full) and are surfaced to
    # the client instead of retried.
    _RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Background task to call external engine and store result in database.

        Timeouts, connection errors, and retryable HTTP statuses are retried
        with exponential backoff until the request succeeds; only
        non-retryable errors mark the future FAILED. The read timeout must
        stay well above the engine's worst-case queue drain: a timeout
        disconnects the request, which makes vLLM abort it, so a too-short
        timeout turns a slow request into an infinite retry loop.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=httpx.Timeout(1800.0, connect=10.0),
                ) as http_client:
                    result = await self._forward_to_engine(
                        sample_req, model_id, checkpoint_id, http_client, base_model=base_model
                    )
                result_data = result.model_dump()
                status = RequestStatus.COMPLETED
                break
            except Exception as e:
                retryable = isinstance(e, (httpx.TimeoutException, httpx.TransportError)) or (
                    isinstance(e, httpx.HTTPStatusError) and e.response.status_code in self._RETRYABLE_STATUS
                )
                if retryable:
                    delay = min(2**attempt, 60)
                    logger.warning(
                        f"External engine request {request_id} attempt {attempt} failed "
                        f"({type(e).__name__}: {e}); retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.exception("External engine error")
                result_data = {"error": f"{type(e).__name__}: {e}", "status": "failed"}
                status = RequestStatus.FAILED
                break

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_to_engine(
        self,
        request,
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
            load_response = await http_client.post(
                "/load_lora_adapter",
                json={"lora_name": model_name, "lora_path": str(target_dir)},
            )
            if load_response.status_code >= 400 and "already" not in load_response.text.lower():
                load_response.raise_for_status()

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

        response = await http_client.post("/completions", json=payload)
        response.raise_for_status()
        result = response.json()

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

        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])
