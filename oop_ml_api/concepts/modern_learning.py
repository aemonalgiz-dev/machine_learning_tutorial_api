"""Bounded teaching fixtures for modern models; all operations live in the SDK."""

from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.numpy.modern import (
    AdversarialGaussian,
    CausalLanguageModel,
    DiagonalGaussian,
    GaussianDiffusion,
    LinearAutoencoder,
    LowRankAdapter,
    SoftmaxPolicy,
    TermOverlapRetriever,
    TokenCrossEntropy,
    TokenDistribution,
)
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/concepts/modern", tags=["modern-learning"])


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Table(BaseModel):
    title: str
    columns: list[str]
    rows: list[list[str | float]]


class View(BaseModel):
    notice: str
    tables: list[Table]
    notes: list[str] = Field(default_factory=list)


def matrix(title: str, rows: list[str], values: np.ndarray) -> Table:
    return Table(
        title=title,
        columns=["Position", *[f"Coordinate {i + 1}" for i in range(values.shape[1])]],
        rows=[[label, *row.tolist()] for label, row in zip(rows, values, strict=True)],
    )


def measures(title: str, values: dict[str, float]) -> Table:
    return Table(
        title=title,
        columns=["Quantity", "Value"],
        rows=[[key, value] for key, value in values.items()],
    )


VOCABULARY = ["<start>", "the", "cat", "dog", "sat", "slept", "<end>"]
PROMPT = [0, 1, 2]


class TransformerRequest(Request):
    animal: Literal["cat", "dog"] = "cat"


@router.post("/transformer-blocks", response_model=View)
def transformer(request: TransformerRequest) -> View:
    tokens = ["<start>", "the", request.animal]
    trace = CausalLanguageModel(len(VOCABULARY)).respond_to(
        np.array([VOCABULARY.index(token) for token in tokens])
    )
    tables = [
        matrix("1. Token embeddings", tokens, trace.embedded),
        matrix("2. Add position information", tokens, trace.positioned),
    ]
    for i, shares in enumerate(trace.attention):
        tables.append(
            Table(
                title=f"3. Attention shares, head {i + 1}",
                columns=["Reading position", *tokens],
                rows=[
                    [token, *row.tolist()]
                    for token, row in zip(tokens, shares, strict=True)
                ],
            )
        )
    tables.extend(
        [
            matrix("4. Add the attention contribution", tokens, trace.after_attention),
            matrix("5. Feed-forward contribution", tokens, trace.feed_forward),
            matrix("6. Add the feed-forward contribution", tokens, trace.hidden),
        ]
    )
    return View(
        notice="One causal block, two heads, four coordinates. Its parameters are "
        "fixed random values, not learned representations.",
        tables=tables,
        notes=[
            (
                "Change the last token. Earlier rows stay unchanged because the "
                "causal mask prevents them from reading it."
            ),
            (
                "Normalization is applied before each branch. The feed-forward "
                "branch uses a wider hidden layer with a ReLU activation."
            ),
        ],
    )


class LossRequest(Request):
    confidence: float = Field(default=1, ge=0, le=4)


def language_loss(confidence: float, response_only: bool = False) -> View:
    # Assigned logits isolate the objective from the untrained decoder.
    contexts = ["<start>", "<start> the", "<start> the cat", "<start> the cat sat"]
    targets = np.array([1, 2, 4, 6])
    logits = np.zeros((4, len(VOCABULARY)))
    logits[np.arange(4), targets] = confidence
    included = np.array([False, False, True, True]) if response_only else None
    loss = TokenCrossEntropy().measure(logits, targets, included)
    return View(
        notice="Assigned vocabulary scores isolate the loss calculation. No "
        "language-model weights are trained in this example.",
        tables=[
            Table(
                title="Score each next-token target",
                columns=[
                    "Visible prefix",
                    "Target",
                    "Target probability",
                    "Token loss",
                    "Scored?",
                ],
                rows=[
                    [
                        context,
                        VOCABULARY[target],
                        float(loss.probabilities[i, target]),
                        float(loss.per_token[i]),
                        "Yes" if loss.included[i] else "No",
                    ]
                    for i, (context, target) in enumerate(
                        zip(contexts, targets, strict=True)
                    )
                ],
            ),
            measures("Objective", {"Mean scored token loss": loss.value}),
            Table(
                title="Derivative at the target score",
                columns=["Target", "Logit derivative"],
                rows=[
                    [VOCABULARY[target], float(loss.gradient[i, target])]
                    for i, target in enumerate(targets)
                ],
            ),
        ],
        notes=[
            (
                "The target is the token after the visible prefix. The target is "
                "never included in its own input."
            ),
            "A zero derivative on a prompt prediction excludes that output "
            "from the objective. Prompt tokens can still provide context."
            if response_only
            else "Increasing the assigned target score raises "
            "its probability and lowers the loss.",
        ],
    )


@router.post("/next-token-prediction", response_model=View)
def next_token(request: LossRequest) -> View:
    return language_loss(request.confidence)


class FineTuningRequest(LossRequest):
    response_only: bool = True


@router.post("/supervised-fine-tuning", response_model=View)
def fine_tuning(request: FineTuningRequest) -> View:
    return language_loss(request.confidence, request.response_only)


class GenerationRequest(Request):
    max_new_tokens: int = Field(default=5, ge=1, le=12)
    temperature: float = Field(default=1, ge=0.1, le=3)
    seed: int = Field(default=0, ge=0, le=100)
    greedy: bool = False


@router.post("/autoregressive-generation", response_model=View)
def generate(request: GenerationRequest) -> View:
    rows = CausalLanguageModel(len(VOCABULARY)).generate(
        np.array(PROMPT),
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        seed=request.seed,
        greedy=request.greedy,
        end_id=6,
    )
    return View(
        notice="This untrained decoder demonstrates the generation loop. Its "
        "outputs are not examples of learned grammar or factual knowledge.",
        tables=[
            Table(
                title="Predict, select, append",
                columns=[
                    "Step",
                    "Prefix read",
                    "Selected token",
                    "Selection probability",
                ],
                rows=[
                    [
                        str(i + 1),
                        " ".join(VOCABULARY[t] for t in row.context),
                        VOCABULARY[row.token_id],
                        float(row.distribution[row.token_id]),
                    ]
                    for i, row in enumerate(rows)
                ],
            ),
            Table(
                title="Distribution at each step",
                columns=["Step", *VOCABULARY],
                rows=[
                    [str(i + 1), *row.distribution.tolist()]
                    for i, row in enumerate(rows)
                ],
            ),
        ],
        notes=[
            (
                "The loop stops when it selects <end> or reaches the requested "
                "length. <start> is an ordinary vocabulary option in this tiny "
                "untrained example and can be selected again."
            ),
            (
                "The model parameters stay fixed. The growing prefix changes "
                "what the next forward pass reads."
            ),
        ],
    )


class SamplingRequest(Request):
    temperature: float = Field(default=1, ge=0.1, le=3)
    top_p: float = Field(default=1, ge=0.05, le=1)


@router.post("/sampling-and-temperature", response_model=View)
def sampling(request: SamplingRequest) -> View:
    logits = np.array([2.0, 1.0, 0.0, -1.0])
    original = TokenDistribution.from_logits(logits, request.temperature)
    filtered = TokenDistribution.from_logits(logits, request.temperature, request.top_p)
    return View(
        notice="These four scores are assigned for comparison. No model "
        "or sampled text is hidden behind the table.",
        tables=[
            Table(
                title="From scores to the selection distribution",
                columns=[
                    "Candidate",
                    "Logit",
                    "After temperature",
                    "Retained?",
                    "Final probability",
                ],
                rows=[
                    [
                        token,
                        float(logits[i]),
                        float(original.probabilities[i]),
                        "Yes" if filtered.retained[i] else "No",
                        float(filtered.probabilities[i]),
                    ]
                    for i, token in enumerate(["sat", "slept", "ran", "sang"])
                ],
            )
        ],
        notes=[
            (
                "Temperature changes the relative shares without changing their "
                "order. Top-p retains the smallest sorted prefix reaching its "
                "threshold, then renormalizes those shares."
            )
        ],
    )


class AutoencoderRequest(Request):
    difference: float = Field(default=0, ge=-2, le=2)


@router.post("/autoencoders", response_model=View)
def autoencoder(request: AutoencoderRequest) -> View:
    inputs = np.array([[1 + request.difference, 1 - request.difference]])
    model = LinearAutoencoder(np.array([[0.5], [0.5]]), np.array([[1.0, 1.0]]))
    result = model.respond_to(inputs)
    return View(
        notice="A chosen averaging encoder compresses two readings into "
        "one. A chosen decoder copies that average back. These "
        "weights have not been fitted.",
        tables=[
            matrix("Input readings", ["Observation"], inputs),
            matrix("Compressed code", ["Observation"], result.code),
            matrix("Reconstructed readings", ["Observation"], result.reconstructed),
            measures(
                "Reconstruction error",
                {"Mean squared error": result.mean_squared_error},
            ),
        ],
        notes=[
            (
                "The slider changes the difference between the readings while "
                "keeping their average fixed. The code cannot preserve that difference."
            )
        ],
    )


class VariationalRequest(Request):
    mean: float = Field(default=0, ge=-2, le=2)
    deviation: float = Field(default=1, ge=0.1, le=2)
    noise: float = Field(default=0.5, ge=-2, le=2)


@router.post("/variational-autoencoders", response_model=View)
def variational(request: VariationalRequest) -> View:
    posterior = DiagonalGaussian(
        np.array([request.mean]), np.array([request.deviation])
    )
    code = posterior.sample(np.array([request.noise]))
    return View(
        notice="Inspect one latent coordinate. The mean, deviation, and "
        "noise draw are chosen inputs. This isolates sampling and "
        "the KL term, without training an encoder or decoder.",
        tables=[
            measures(
                "Reparameterized sample",
                {
                    "Mean": request.mean,
                    "Standard deviation": request.deviation,
                    "Noise draw": request.noise,
                    "Sampled code": float(code[0]),
                },
            ),
            measures(
                "Distance from the standard normal prior",
                {"KL divergence": posterior.kl_to_standard_normal},
            ),
        ],
        notes=[
            (
                "Changing the noise draw changes the sampled code, but does not "
                "change the distribution's KL divergence. A full VAE also needs "
                "a reconstruction likelihood and gradients through its encoder and decoder."
            )
        ],
    )


class GanRequest(Request):
    steps: int = Field(default=30, ge=0, le=150)
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.3)


@router.post("/generative-adversarial-networks", response_model=View)
def gan(request: GanRequest) -> View:
    noise = np.random.default_rng(3).normal(size=128)
    real = noise + 1.5
    history = AdversarialGaussian().train(
        real, noise, request.steps, request.learning_rate
    )
    checkpoints = sorted({0, len(history) // 2, len(history) - 1})
    return View(
        notice="An actual, limited GAN update: the generator learns only "
        "a shift of fixed Gaussian noise. The discriminator is a "
        "single sigmoid unit. This is not an image generator.",
        tables=[
            Table(
                title="Alternating training checkpoints",
                columns=[
                    "Step",
                    "Generator shift",
                    "Mean D(real)",
                    "Mean D(fake)",
                    "D loss",
                    "G loss",
                ],
                rows=[
                    [
                        str(i),
                        history[i].generator_mean,
                        history[i].real_score,
                        history[i].fake_score,
                        history[i].discriminator_loss,
                        history[i].generator_loss,
                    ]
                    for i in checkpoints
                ],
            ),
            measures(
                "Target and final parameters",
                {
                    "Target shift": 1.5,
                    "Learned generator shift": history[-1].generator_mean,
                    "Discriminator weight": history[-1].discriminator_weight,
                    "Discriminator bias": history[-1].discriminator_bias,
                },
            ),
        ],
        notes=[
            (
                "Each step updates the discriminator with the generator fixed, "
                "then the generator with the updated discriminator fixed."
            ),
            (
                "The two losses belong to different objectives. Either can rise "
                "during training; they are not a single progress score."
            ),
        ],
    )


class DiffusionRequest(Request):
    step: int = Field(default=5, ge=0, le=20)
    estimate_fraction: float = Field(default=0, ge=0, le=1)


@router.post("/diffusion-models", response_model=View)
def diffusion(request: DiffusionRequest) -> View:
    model = GaussianDiffusion(np.full(20, 0.1))
    clean = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    noise = np.random.default_rng(4).normal(size=6)
    noisy = model.add_noise(clean, noise, request.step)
    recovered = model.recover_clean(
        noisy, request.estimate_fraction * noise, request.step
    )
    return View(
        notice="A six-pixel signal and known Gaussian noise expose the "
        "forward process. The oracle fraction scales the true noise; "
        "it is not a trained noise predictor or a reverse sampler.",
        tables=[
            Table(
                title="Follow each pixel",
                columns=[
                    "Pixel",
                    "Clean",
                    "Added noise draw",
                    "Noisy signal",
                    "Estimated clean signal",
                ],
                rows=[
                    [
                        str(i + 1),
                        float(clean[i]),
                        float(noise[i]),
                        float(noisy[i]),
                        float(recovered[i]),
                    ]
                    for i in range(len(clean))
                ],
            ),
            measures(
                "Signal retention",
                {"Cumulative alpha": float(model.retention[request.step])},
            ),
        ],
        notes=[
            (
                "At step zero, the signal is unchanged. At oracle fraction one, "
                "the exact noise is supplied, so the algebra recovers the clean signal."
            ),
            (
                "Generation does not have access to that exact noise. It requires "
                "a trained predictor and a sequence of reverse updates."
            ),
        ],
    )


class LoraRequest(Request):
    rank: int = Field(default=1, ge=1, le=3)
    alpha: float = Field(default=1, ge=0.1, le=4)


@router.post("/low-rank-adaptation", response_model=View)
def lora(request: LoraRequest) -> View:
    rng = np.random.default_rng(9)
    down = rng.normal(0, 0.3, (4, 3))[:, : request.rank]
    up = rng.normal(0, 0.3, (3, 4))[: request.rank]
    model = LowRankAdapter(np.eye(4), down, up, request.alpha)
    inputs = np.array([[1.0, 0.0, 0.5, -0.5]])
    return View(
        notice="A frozen identity projection and chosen adapter factors "
        "show how the update is assembled. The factors are not fitted.",
        tables=[
            matrix(
                "Low-rank weight update",
                [f"Input {i + 1}" for i in range(4)],
                model.delta,
            ),
            matrix("Original output through the identity", ["Observation"], inputs),
            matrix("Output with adapter", ["Observation"], model.respond_to(inputs)),
            measures(
                "Parameter counts",
                {
                    "Full projection parameters": float(model.base.size),
                    "Adapter parameters": float(model.trainable_parameters),
                },
            ),
        ],
        notes=[
            (
                "For this tiny matrix, rank two uses as many parameters as a full "
                "update and rank three uses more. LoRA saves trainable parameters "
                "when the selected rank is small enough relative to both widths."
            )
        ],
    )


PASSAGES = [
    "The library opens at nine on weekdays. It closes at six.",
    "The library opens at ten on Saturday. It closes at four.",
    "Borrowed books may be renewed online using your library account.",
]


class RetrievalRequest(Request):
    query: str = Field(
        default="library Saturday", min_length=1, max_length=160, pattern=r"\S"
    )
    limit: int = Field(default=2, ge=1, le=3)


@router.post("/retrieval-augmented-generation", response_model=View)
def retrieve(request: RetrievalRequest) -> View:
    passages = TermOverlapRetriever(PASSAGES).retrieve(request.query, request.limit)
    return View(
        notice="The example runs exact word-count retrieval over three "
        "invented library passages. It shows the context to supply "
        "to a model; it does not call a language model.",
        tables=[
            Table(
                title="Source collection",
                columns=["Source", "Passage"],
                rows=[[str(i + 1), p] for i, p in enumerate(PASSAGES)],
            ),
            Table(
                title="Retrieved context",
                columns=["Source", "Cosine score", "Passage"],
                rows=[[str(p.index + 1), p.score, p.text] for p in passages],
            ),
        ],
        notes=[
            (
                "A query with no indexed words returns no passages. Word overlap "
                "cannot recognize synonyms or guarantee that a passage answers the question."
            ),
            (
                "In a complete RAG system, the question and selected passages "
                "enter a prompt, and the generated answer is checked against its cited sources."
            ),
        ],
    )


class ReinforcementRequest(Request):
    episodes: int = Field(default=60, ge=0, le=200)
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    seed: int = Field(default=0, ge=0, le=100)


@router.post("/reinforcement-learning", response_model=View)
def reinforcement(request: ReinforcementRequest) -> View:
    actions = ["Left", "Straight", "Right"]
    outcomes = ["Slow delivery", "Quick delivery", "Blocked route"]
    rewards = np.array([0.2, 1.0, -1.0])
    run = SoftmaxPolicy(np.zeros(3)).train_bandit(
        rewards, request.episodes, request.learning_rate, request.seed
    )
    shown = list(enumerate(run.updates, start=1))
    if len(shown) > 6:
        shown = [*shown[:5], shown[-1]]
    return View(
        notice="A trained one-state policy in a simulated one-step environment. "
        "Rewards are assigned for this lesson. Each update receives only the "
        "chosen route's reward; no language model or navigation network is trained.",
        tables=[
            Table(
                title="1. The environment and the learned policy",
                columns=[
                    "Action",
                    "Outcome",
                    "Assigned reward",
                    "Initial probability",
                    "Final probability",
                    "Times sampled",
                ],
                rows=[
                    [
                        action,
                        outcomes[i],
                        float(rewards[i]),
                        float(run.initial[i]),
                        float(run.final[i]),
                        int(run.counts[i]),
                    ]
                    for i, action in enumerate(actions)
                ],
            ),
            Table(
                title="2. Sampled feedback, first five episodes and final episode",
                columns=[
                    "Episode",
                    "Sampled action",
                    "Observed reward",
                    "Probability before",
                    "Probability after",
                ],
                rows=[
                    [
                        episode,
                        actions[row.action],
                        row.return_value,
                        float(row.before[row.action]),
                        float(row.after[row.action]),
                    ]
                    for episode, row in shown
                ],
            ),
        ],
        notes=[
            "Set episodes to zero to see the equal starting probabilities and no updates.",
            "A positive reward increases the chosen action's probability; a negative reward decreases it. The baseline is zero here.",
            "Every control change starts a fresh run. Compare seeds to see different experiences, then raise episodes to collect more feedback.",
            "The payoff table is visible to you and the simulator. The learning rule never receives rewards for unchosen actions in that episode.",
        ],
    )


class EvaluationRequest(Request):
    confidence: float = Field(default=1, ge=0, le=4)
    wrong: bool = False


@router.post("/evaluating-generative-models", response_model=View)
def evaluate(request: EvaluationRequest) -> View:
    targets = np.array([0, 1, 2])
    logits = np.zeros((3, 3))
    logits[np.arange(3), (targets + 1) % 3 if request.wrong else targets] = (
        request.confidence
    )
    result = TokenCrossEntropy().measure(logits, targets)
    return View(
        notice="Assigned scores on three target tokens isolate held-out "
        "language-model scoring. These are not benchmark results "
        "from a trained model.",
        tables=[
            Table(
                title="Probability assigned to the observed continuation",
                columns=["Target", "Probability", "Negative log probability"],
                rows=[
                    [
                        str(t),
                        float(result.probabilities[i, t]),
                        float(result.per_token[i]),
                    ]
                    for i, t in enumerate(targets)
                ],
            ),
            measures(
                "Aggregate token metrics",
                {
                    "Mean negative log likelihood": result.value,
                    "Perplexity": result.perplexity,
                },
            ),
        ],
        notes=[
            (
                "Try boosting the wrong token. Greater confidence then increases "
                "loss and perplexity. At uniform scores, perplexity equals the "
                "three-token vocabulary size."
            ),
            (
                "These metrics measure continuation probability. They do not "
                "establish factual accuracy, source support, or usefulness."
            ),
        ],
    )
