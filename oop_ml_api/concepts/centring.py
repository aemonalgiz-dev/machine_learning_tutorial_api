"""Centring on the mean, the half of standardising that is not about scale.

Every endpoint here is registered under ``/concepts/centring-on-the-mean``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about subtracting the mean alone, which methods require it, and what goes silently wrong when it is skipped.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(
    prefix="/concepts/centring-on-the-mean", tags=["centring-on-the-mean"]
)
