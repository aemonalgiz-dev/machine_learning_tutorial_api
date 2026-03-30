"""The application: mount the concept routers, wire CORS and error handling.

Deliberately thin. It creates the app, lets the website's origin call it, turns
library failures into typed 422s, and mounts one router per concept. Everything
with any mathematics in it lives in the library; everything with a request shape
in it lives in a concept module. This file only assembles them.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from oop_ml_api.concepts import (
    decision_tree,
    ensembles,
    k_nearest,
    logistic_regression,
    penalised_regression,
    polynomial_regression,
    simple_linear_regression,
)
from oop_ml_api.errors import install_error_handling
from oop_ml_api.primers import calculus, linear_algebra, statistics

CONCEPT_ROUTERS = (
    simple_linear_regression.router,
    polynomial_regression.router,
    penalised_regression.router,
    logistic_regression.router,
    k_nearest.router,
    decision_tree.router,
    ensembles.router,
)
"""Every concept the API serves. Adding one is appending its router here."""

PRIMER_ROUTERS = (calculus.router, linear_algebra.router, statistics.router)
"""The foundation-maths primers, kept separate because they are not models."""


def allowed_origins() -> list[str]:
    """Which web origins may call this API.

    Read from ``OOP_ML_API_ORIGINS`` (comma-separated) so the deployed website's
    origin is configuration, not code, and falls back to the local Next.js dev
    server. A browser will not let the site call the API cross-origin without
    this, so it is the one piece of web plumbing the API cannot skip.
    """
    configured = os.environ.get("OOP_ML_API_ORIGINS")

    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]

    return ["http://localhost:3000", "http://127.0.0.1:3000"]


def create_app() -> FastAPI:
    """Build the application. A factory so tests get a fresh instance."""
    app = FastAPI(
        title="oop_ml teaching API",
        version="0.1.0",
        summary="Small numeric inputs in, fitted models out, for the teaching site.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    install_error_handling(app)

    for router in (*CONCEPT_ROUTERS, *PRIMER_ROUTERS):
        app.include_router(router)

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """A liveness check that touches no library code."""
        return {"status": "ok"}

    return app


app = create_app()
