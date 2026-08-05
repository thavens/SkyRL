"""Shared test utilities and fixtures for tinker tests."""

import asyncio
import time
from typing import Callable


def wait_for_condition(
    condition_fn: Callable[[], bool],
    timeout_sec: float = 10,
    poll_interval_sec: float = 0.1,
) -> bool:
    """Poll until condition_fn returns True or timeout is reached. Returns True if condition was met."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if condition_fn():
            return True
        time.sleep(poll_interval_sec)
    return False


def unload_model(base_url: str, model_id: str, api_key: str = "tml-dummy"):
    """Unload a model, blocking until the server confirms it and returning the response.

    Adapter slots are in-process engine state that a departing client does not release, so
    anything sharing a server has to hand them back explicitly. ``models.unload`` is the
    public deletion path Tinker exposes.
    """
    # Imported lazily: this conftest is collected for every test under tests/tinker, including
    # modules that importorskip("tinker") because the extra may not be installed.
    import tinker
    from tinker import types

    async def run():
        async with tinker._client.AsyncTinker(api_key=api_key, base_url=base_url) as client:  # type: ignore[attr-defined]
            future = await client.models.unload(request=types.UnloadModelRequest(model_id=model_id))
            while True:
                result = await client.futures.retrieve(
                    request=types.FutureRetrieveRequest(request_id=future.request_id)
                )
                if isinstance(result, types.UnloadModelResponse):
                    return result
                await asyncio.sleep(0.1)

    return asyncio.run(run())
