"""Convolutional networks, assembled from the layers.

Every endpoint here is registered under ``/concepts/convolutional-networks``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about convolution, pooling and flattening stacked into a model that reads a picture, and what each part contributes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(
    prefix="/concepts/convolutional-networks", tags=["convolutional-networks"]
)
