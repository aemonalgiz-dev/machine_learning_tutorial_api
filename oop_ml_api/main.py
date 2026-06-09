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
    backpropagation,
    byte_pair_encoding,
    bytes_and_characters,
    centring,
    codebook_quantisation,
    convolution,
    convolutional_networks,
    decision_tree,
    dense_layers,
    distance_and_similarity,
    distance_metrics,
    dropout,
    ensembles,
    evaluation,
    fasttext,
    feature_importance,
    feature_scaling,
    filters_and_edges,
    finite_scalar_quantisation,
    finite_state_morphology,
    glove,
    gradient_descent_regression,
    greedy_coverage,
    grid_search,
    haar_cascades,
    hashing_characters,
    hebbian_principal_components,
    hidden_markov_words,
    hopfield_network,
    judging_a_classifier,
    k_nearest,
    kernel_pca,
    kernel_ridge,
    kernel_trick,
    keypoints_and_descriptors,
    latent_semantic_analysis,
    logistic_regression,
    loss_functions,
    maximum_matching,
    morfessor,
    moses_words,
    multiclass_classification,
    mutual_information,
    n_grams,
    neurons,
    normalisation_layers,
    oriented_gradients,
    paragraph_vectors,
    patching,
    pattern_words,
    penalised_regression,
    pipelines,
    pointwise_words,
    polynomial_features,
    polynomial_regression,
    pooling,
    pooling_a_text,
    principal_components,
    random_indexing,
    restricted_boltzmann_machine,
    self_organising_map,
    sentencepiece,
    shapes,
    simple_linear_regression,
    standard_score,
    template_matching,
    token_basics,
    training_a_network,
    treebank_words,
    unicode_words,
    unigram_model,
    unsupervised,
    vocabulary_transfer,
    whitespace_words,
    word2vec,
    word_lattice,
    word_vectors,
    wordpiece,
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
    unsupervised.router,
    kernel_trick.router,
    evaluation.router,
    feature_scaling.router,
    normalisation_layers.router,
    hopfield_network.router,
    restricted_boltzmann_machine.router,
    self_organising_map.router,
    hebbian_principal_components.router,
    gradient_descent_regression.router,
    multiclass_classification.router,
    distance_metrics.router,
    kernel_ridge.router,
    kernel_pca.router,
    judging_a_classifier.router,
    grid_search.router,
    pipelines.router,
    feature_importance.router,
    neurons.router,
    dense_layers.router,
    loss_functions.router,
    backpropagation.router,
    convolution.router,
    pooling.router,
    shapes.router,
    dropout.router,
    training_a_network.router,
    byte_pair_encoding.router,
    bytes_and_characters.router,
    centring.router,
    codebook_quantisation.router,
    convolutional_networks.router,
    distance_and_similarity.router,
    fasttext.router,
    filters_and_edges.router,
    finite_scalar_quantisation.router,
    finite_state_morphology.router,
    glove.router,
    greedy_coverage.router,
    haar_cascades.router,
    hashing_characters.router,
    hidden_markov_words.router,
    keypoints_and_descriptors.router,
    latent_semantic_analysis.router,
    maximum_matching.router,
    morfessor.router,
    moses_words.router,
    mutual_information.router,
    n_grams.router,
    oriented_gradients.router,
    paragraph_vectors.router,
    patching.router,
    pattern_words.router,
    pointwise_words.router,
    polynomial_features.router,
    pooling_a_text.router,
    principal_components.router,
    random_indexing.router,
    sentencepiece.router,
    standard_score.router,
    template_matching.router,
    token_basics.router,
    treebank_words.router,
    unicode_words.router,
    unigram_model.router,
    vocabulary_transfer.router,
    whitespace_words.router,
    word2vec.router,
    word_lattice.router,
    word_vectors.router,
    wordpiece.router,
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
