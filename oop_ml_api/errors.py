"""Turning the library's typed failures into HTTP the website can act on.

The whole point of ``MLLibError`` being one hierarchy -- the thing the library's
serving-readiness audit spent its effort on -- is realised here: a single
handler catches everything the library can raise and answers ``422 Unprocessable
Entity`` with the typed message, because every such failure is the *caller's*
data being wrong, not the server breaking. Anything that is **not** an
``MLLibError`` is a genuine 500: a bug in this layer or the library, and it is
logged rather than explained to the client, because its message is for the
developer, not the learner.

That split is only trustworthy because the audit closed every path by which a
bare ``numpy`` or ``ValueError`` used to escape. If one still did, it would land
here as a 500 -- visible, not silently a 422 -- which is the right failure of the
two.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from oop_ml.core.exceptions import MLLibError

logger = logging.getLogger("oop_ml_api")


def install_error_handling(app: FastAPI) -> None:
    """Register the two handlers that map failures to responses.

    Called once by the app factory. Kept here rather than inline so the mapping
    -- library error is the client's fault, anything else is ours -- lives in
    one readable place.
    """

    @app.exception_handler(MLLibError)
    async def _library_error(request: Request, error: MLLibError) -> JSONResponse:
        """A typed library refusal: the request's data cannot be computed on.

        422 rather than 400, because the request was well formed as JSON and
        the values are individually valid -- it is their *meaning* the library
        rejects (too few points, a singular system, classes that leave a gap).
        The message is the library's own, written to be read.
        """
        return JSONResponse(
            status_code=422,
            content={
                "error": type(error).__name__,
                "detail": str(error),
            },
        )

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, error: Exception) -> JSONResponse:
        """Anything the library did not raise on purpose: a real server error.

        Logged with its traceback for the developer and answered with a plain
        message, because an unexpected error's detail is an implementation leak,
        not something a learner should see or could act on.
        """
        logger.exception("unhandled error serving %s", request.url.path)

        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "detail": "the server failed to compute this request",
            },
        )
