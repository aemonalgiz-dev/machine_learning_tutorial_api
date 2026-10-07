"""Compute the slow answers before anybody asks for them.

A handful of endpoints fit something substantial the first time they are
called: a word2vec model over three chapters of a book, a greedy vocabulary
over a capped corpus, the picture networks the picture pages train. Each
caches its answer for the life of the process, so only the first visitor
waits, but that visitor can wait fifteen seconds, and a widget that gives up
first shows a failure for a page that is working.

So the application asks itself for them, once, on a background thread as it
starts. A visitor who arrives before the thread reaches their request waits
exactly as they would have, and everyone after that is answered from the
cache. The thread is a daemon, so it never holds a shutting-down server open,
and a failure in one request is logged and does not stop the others.

Each request goes through the application as a real one would, routing, query
parsing and validation included, by calling it as an ASGI application with a
scope built here. That needs nothing beyond the standard library, where a test
client would need a package the application does not otherwise depend on, and
it means a request listed here with a query string is warmed with exactly the
arguments a page will send.

The list is by URL rather than by function, because the URL is what a page
asks for and what a report about a slow page names. A path here that no longer
matches a GET route is a mistake the test module catches.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Iterable, Iterator

from fastapi import FastAPI
from fastapi.routing import APIRoute

# Uvicorn configures its own loggers and leaves the root one alone, so a module
# logger's INFO lines never reach the console the server was started from.
# Writing through uvicorn's error logger puts "warmed ... in 4.21 s" beside the
# server's own start-up lines, which is the one place anybody would look for it.
logger = logging.getLogger("uvicorn.error")

_NETWORK_VARIANTS = ("reference", "without_pooling", "one_round", "dense")
_NETWORK_SEEDS = (0, 1, 2)

WARM_REQUESTS = (
    "/concepts/greedy-coverage/capping",
    "/concepts/image-alignment/turn-cost",
    "/concepts/image-alignment/orientation",
    "/concepts/image-alignment/describe",
    "/concepts/image-alignment/ransac",
    "/concepts/image-alignment/pyramid",
    "/concepts/learning-boundaries-from-examples/comparison",
    "/concepts/word2vec/book-space",
    "/concepts/segmenting-with-a-hidden-model/scale",
    *(
        f"/concepts/convolutional-networks/network?variant={variant}&seed={seed}"
        for variant in _NETWORK_VARIANTS
        for seed in _NETWORK_SEEDS
    ),
    "/concepts/convolutional-networks/maps",
    "/concepts/convolutional-networks/unseen",
    "/concepts/a-vector-for-a-picture/walk",
    "/concepts/a-vector-for-a-picture/sphere",
    "/concepts/a-vector-for-a-picture/edges",
    "/concepts/a-vector-for-a-picture/seed/0",
    "/concepts/a-vector-for-a-picture/seed/1",
    "/concepts/a-vector-for-a-picture/seed/2",
    "/concepts/searching-a-collection-of-pictures/overview",
    "/concepts/searching-a-collection-of-pictures/map",
    "/concepts/searching-a-collection-of-pictures/trade-off",
    "/concepts/searching-a-collection-of-pictures/hashing",
    "/concepts/searching-a-collection-of-pictures/concentration",
    "/concepts/searching-a-collection-of-pictures/seeds/0",
    "/concepts/searching-a-collection-of-pictures/seeds/1",
    "/concepts/searching-a-collection-of-pictures/seeds/2",
    "/concepts/centring-on-the-mean/crowd",
    *(
        f"/concepts/u-net/network?skip={skip}&seed={seed}"
        for skip in ("true", "false")
        for seed in _NETWORK_SEEDS
    ),
    "/concepts/u-net/baselines",
    "/concepts/u-net/gradient-check",
    "/concepts/u-net/larger",
    *(
        f"/concepts/learning-the-metric-itself/{endpoint}"
        for endpoint in (
            "comparison",
            "comparison/seed-1",
            "comparison/seed-2",
            "comparison/without-cross",
            "comparison/without-disc",
            "comparison/without-square",
            "sphere",
            "worked-pair",
            "shared-weights",
            "without-different-kinds",
            "cost",
            "wider-positions",
            "margin?margin=2&rate_rule=fixed&seed=0",
        )
    ),
)
"""The GET requests worth answering at start, in the order they are made."""

ENVIRONMENT_SWITCH = "OOP_ML_WARM_ON_START"
"""Set to ``0`` to start without warming, for a quick restart while editing."""


def warming_enabled() -> bool:
    """Whether the application should warm its slow requests as it starts."""
    return os.environ.get(ENVIRONMENT_SWITCH, "1") != "0"


def _every_route(routes: Iterable[object]) -> Iterator[APIRoute]:
    """Every endpoint route, looking inside the routers the app included.

    FastAPI keeps an included router as one wrapper holding the original
    router rather than copying its routes onto the app, so a flat scan of
    ``app.routes`` finds none of the concept endpoints. Each concept router
    carries its own full prefix and is included without another, so a route's
    ``path`` inside it is already the path a page asks for.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _every_route(inner.routes)


def serves_get(app: FastAPI, request: str) -> bool:
    """Whether some GET route matches ``request``, path parameters included.

    Matched against each route's own compiled pattern rather than its path
    string, so ``/seeds/2`` is recognised as the route ``/seeds/{weight_seed}``.
    """
    path = request.partition("?")[0]
    return any(
        "GET" in route.methods and route.path_regex.match(path) is not None
        for route in _every_route(app.routes)
    )


async def _status_of_get(app: FastAPI, request: str) -> int:
    """Send one GET through the application and return the status it answered."""
    path, _, query = request.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(b"host", b"warmup")],
        "client": ("127.0.0.1", 0),
        "server": ("warmup", 80),
    }
    status = 0

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = int(message["status"])  # type: ignore[arg-type]

    await app(scope, receive, send)  # type: ignore[arg-type]
    return status


def warm(app: FastAPI) -> None:
    """Make every request in :data:`WARM_REQUESTS` once, logging each outcome."""
    for request in WARM_REQUESTS:
        started = time.perf_counter()
        try:
            status = asyncio.run(_status_of_get(app, request))
        except Exception:
            # One failure must not stop the rest, and the log says which.
            logger.exception("warming %s failed", request)
            continue
        elapsed = time.perf_counter() - started
        if status == 200:
            logger.info("warmed %s in %.2f s", request, elapsed)
        else:
            logger.warning("warming %s answered %d", request, status)


def warm_in_background(app: FastAPI) -> threading.Thread:
    """Start :func:`warm` on a daemon thread and hand the thread back."""
    thread = threading.Thread(
        target=warm, args=(app,), name="warm-slow-requests", daemon=True
    )
    thread.start()
    return thread
