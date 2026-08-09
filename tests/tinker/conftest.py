"""Shared test utilities and fixtures for tinker tests."""

import asyncio
import time
import urllib.error
import urllib.request
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


def api_server_is_up(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


def unload_model(base_url: str, model_id: str, api_key: str = "tml-dummy") -> dict:
    """Unload a model, blocking until the server confirms it, and return the raw response body.

    Adapter slots are in-process engine state that a departing client does not release, so
    anything sharing a server has to hand them back explicitly. ``models.unload`` is the
    public deletion path Tinker exposes.

    ``futures.retrieve`` casts to the ``FutureRetrieveResponse`` union, which has no
    discriminator. tinker>=0.17 resolves such a union to its first member
    (``TryAgainResponse``) regardless of the payload, so ``isinstance(result,
    UnloadModelResponse)`` is never true and polling on it spins forever. Read the raw body
    instead and dispatch on the server's ``type`` field. (The high-level
    ``APIFuture.result()`` path is unaffected; it deserializes into a concrete type rather
    than the union.)
    """
    # Imported lazily: this conftest is collected for every test under tests/tinker, including
    # modules that importorskip("tinker") because the extra may not be installed.
    import tinker
    from tinker import types

    async def run() -> dict:
        async with tinker._client.AsyncTinker(api_key=api_key, base_url=base_url) as client:  # type: ignore[attr-defined]
            future = await client.models.unload(request=types.UnloadModelRequest(model_id=model_id))
            while True:
                response = await client.futures.with_raw_response.retrieve(
                    request=types.FutureRetrieveRequest(request_id=future.request_id)
                )
                body = await response.json()
                if body.get("type") == "try_again":
                    await asyncio.sleep(0.1)
                    continue
                return body

    return asyncio.run(run())
