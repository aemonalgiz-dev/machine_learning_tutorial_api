# oop_ml teaching API

A thin HTTP layer over the [`oop_ml`](../) library. The teaching website computes
against it: small numeric inputs arrive as JSON, become `oop_ml` objects and a
model, and the fitted answer goes back as JSON. **No mathematics lives here** —
the library owns all of it, and this layer is a translator plus a bouncer at the
door.

## Why it is a separate repository

Three components, three repositories, each deploying on its own cadence:

- **`oop_ml`** — the library (the parent directory).
- **this API** — a stateless HTTP service that imports the library.
- **the website** — a React/Next.js frontend that calls this API over HTTP and
  never imports the library.

The website talks to the API over HTTP and the API imports the library, so
nothing is coupled tighter than a versioned contract — which is what lets the
three repositories stay independent without submodules.

## Design

- **CPU-bound endpoints are plain `def`, not `async def`.** FastAPI runs a
  synchronous path operation in its worker threadpool, off the event loop, which
  is exactly what a CPU-bound numpy library wants behind a web server. An
  `async def` would run the fit *on* the event loop and block every other
  request for its duration.
- **Every failure the library can raise is one hierarchy (`MLLibError`), mapped
  to `422`.** Such a failure is the caller's data being wrong, not the server
  breaking. Anything else is a genuine `500`, logged rather than explained. The
  library's serving-readiness audit is what makes that split trustworthy — it
  closed every path by which a bare `numpy` or `ValueError` used to escape.
- **Inputs are small by design and bounded at the door** (`limits.py`). A
  teaching example is a handful of points; a request outside the bounds is a
  typed `422` with the limit named, never a slow compute.
- **A concept is a module** under `oop_ml_api/concepts/`, one router each. Adding
  a concept is adding a module and mounting its router.

## Running

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]" -e ..
.venv/Scripts/uvicorn oop_ml_api.main:app --reload
```

The dev server listens on `http://localhost:8000`; interactive docs are at
`/docs`. The website's origin is configured through `OOP_ML_API_ORIGINS`
(comma-separated), defaulting to the local Next.js dev server.

## Tests

```bash
.venv/Scripts/pytest
```

They go through the real ASGI app with a test client, exercising request
parsing, the library call, and the response shape together — including the
input-limit refusals and the library-error-to-422 mapping, which are the point
of the layer.

## Endpoints

| method | path | concept |
| --- | --- | --- |
| `GET` | `/health` | liveness, touches no library code |
| `POST` | `/concepts/simple-linear-regression/fit` | fit a line to points; returns slope, intercept, R², the fitted line, and per-point residuals |
| `GET` | `/docs` | interactive OpenAPI documentation |
