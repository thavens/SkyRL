"""Least-connections HTTP load balancer for the Qwen3.6-27B defender replicas.

Exists because `VLLMSamplerConfig.base_url` is a single string, so the training
loop cannot address two vLLM instances itself. This binds the port the client
already points at and spreads requests over the replicas behind it.

    uv run --isolated --with aiohttp python tools/defender_lb.py \
        --port 8002 --backend http://127.0.0.1:8003 --backend http://127.0.0.1:8004

Least connections rather than round robin: defender requests differ several-fold
in cost (prompt length and completion length both vary), so round robin would
queue work behind a slow replica while the other idles.

Localhost only, matching the servers it fronts -- neither replica has auth.
"""

import argparse
import asyncio
import itertools
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

# Headers that describe a single hop and must not be forwarded.
HOP_BY_HOP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}


class Balancer:
    def __init__(self, backends):
        self.backends = list(backends)
        self.inflight = {b: 0 for b in self.backends}
        self.served = {b: 0 for b in self.backends}
        self.failed = {b: 0 for b in self.backends}
        self._rr = itertools.count()

    def order(self):
        """Backends by (in-flight, rotating tiebreak) -- least loaded first."""
        start = next(self._rr) % len(self.backends)
        rotated = self.backends[start:] + self.backends[:start]
        return sorted(rotated, key=lambda b: self.inflight[b])

    def stats(self):
        return {
            b: {"inflight": self.inflight[b], "served": self.served[b], "failed": self.failed[b]} for b in self.backends
        }


async def handle(request: web.Request) -> web.StreamResponse:
    bal: Balancer = request.app["bal"]
    session: ClientSession = request.app["session"]

    if request.path == "/lb_stats":
        return web.json_response({"uptime_s": round(time.time() - request.app["t0"], 1), "backends": bal.stats()})

    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}

    last_error = None
    # Retry on a different replica only while no bytes have reached the client:
    # once the response has started, the request is no longer safe to replay.
    # `started` flips once prepare() has put status+headers on the wire, which is
    # the point of no return -- a mid-stream failure after that must propagate,
    # because re-preparing a second StreamResponse on an already-started request
    # raises inside aiohttp and the client sees a truncated body plus a hard
    # connection error instead of the 502 this fall-through is meant to produce.
    started = False
    for backend in bal.order():
        url = backend + request.path_qs
        bal.inflight[backend] += 1
        try:
            async with session.request(
                request.method, url, data=body, headers=headers, allow_redirects=False
            ) as upstream:
                out = web.StreamResponse(status=upstream.status)
                for k, v in upstream.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        out.headers[k] = v
                await out.prepare(request)
                started = True
                async for chunk in upstream.content.iter_chunked(65536):
                    await out.write(chunk)
                await out.write_eof()
                bal.served[backend] += 1
                return out
        except (OSError, asyncio.TimeoutError) as e:
            bal.failed[backend] += 1
            last_error = f"{backend}: {type(e).__name__}: {e}"
            if started:
                # Response already on the wire: cannot fail over, cannot send a 502.
                # Dropping the connection is the only honest signal left, and it
                # is what lets the client tell a truncated answer from a complete one.
                raise
        finally:
            bal.inflight[backend] -= 1

    return web.json_response(
        {"error": {"message": f"all defender replicas failed ({last_error})", "type": "bad_gateway"}},
        status=502,
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", action="append", required=True)
    # Generation can legitimately take many minutes; the client's own timeout is
    # 900 s, so the balancer must not be the one to give up first.
    ap.add_argument("--sock-read-timeout", type=float, default=1800.0)
    # Must exceed the client's max_concurrency (128 in qwen36_clap.py) with headroom.
    ap.add_argument("--max-connections", type=int, default=512)
    args = ap.parse_args()

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["bal"] = Balancer(args.backend)
    app["t0"] = time.time()
    # aiohttp's default connector caps at 100 total connections. The client drives
    # 128 concurrent defender requests, so the default silently queues the excess
    # until they trip the connect timeout -- observed as ConnectionTimeoutError on
    # 24 of 158 requests. Size the pool above the client's max_concurrency.
    app["session"] = ClientSession(
        connector=TCPConnector(limit=args.max_connections, limit_per_host=args.max_connections),
        timeout=ClientTimeout(total=None, connect=120, sock_connect=60, sock_read=args.sock_read_timeout),
    )
    app.router.add_route("*", "/{tail:.*}", handle)

    async def close_session(app):
        await app["session"].close()

    app.on_cleanup.append(close_session)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print(f"defender_lb on http://{args.host}:{args.port} -> {', '.join(args.backend)}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
