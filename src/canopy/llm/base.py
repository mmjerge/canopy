"""Provider-agnostic LLM client interface and helpers.

The routing / prompt-trimming experiments only need a uniform way to (a) generate a
completion and (b) know its token cost. :class:`LLMClient` is that contract; concrete
providers (Bedrock, OpenAI, ...) implement it, and everything downstream
(:func:`measure_quality_matrix`, the reasoning-search harness) depends on the Protocol
rather than any specific SDK.

A generation returns ``(text, input_tokens, output_tokens)`` so token-accurate cost can be
computed via :meth:`LLMClient.price_per_1k`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# A provider returns the completion text plus input/output token counts.
Generation = tuple[str, int, int]

# The reasoning-search harness (``canopy.bandits.reasoning_llm``) consumes this shape.
GenerateFn = Callable[[str, int, int], str]


@runtime_checkable
class LLMClient(Protocol):
    """Uniform interface over a chat/completions model provider."""

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        """Return ``(text, input_tokens, output_tokens)`` for one prompt.

        ``seed`` distinguishes independent samples of the *same* prompt (e.g. self-consistency
        or the branching candidates of value-guided search). Providers sample stochastically at
        ``temperature`` regardless, but ``seed`` must reach any response cache so that repeated
        same-prompt draws are kept as distinct samples rather than collapsing to one cached
        completion (which would silently defeat sampling-based methods).
        """
        ...

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        """Return ``(input_price, output_price)`` per 1K tokens in USD for ``model_id``."""
        ...


def as_generate_fn(
    client: LLMClient,
    model_id: str,
    temperature: float = 0.7,
) -> GenerateFn:
    """Adapt an :class:`LLMClient` to the reasoning harness's ``(prompt, max_tokens, seed)``.

    The reasoning-search strategies in :mod:`canopy.bandits.reasoning_llm` take a plain
    ``generate(prompt, max_tokens, seed) -> text`` callable. The ``seed`` is forwarded so that a
    response cache keys each independent same-prompt sample separately: providers sample at
    ``temperature``, but without a per-sample key a cache would serve the first draw for every
    call, collapsing the ``branching`` candidates and best-of-N samples to one and silently
    defeating the search / self-consistency being measured.
    """

    def generate(prompt: str, max_tokens: int, seed: int) -> str:
        text, _, _ = client.generate(
            model_id, prompt, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
        return text

    return generate


def measure_quality_matrix(
    prompts: Sequence[str],
    model_ids: Sequence[str],
    client: LLMClient,
    grade: Callable[[str, str], float],
    pricing: dict[str, tuple[float, float]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build ``(quality, costs)`` for :class:`~canopy.bandits.routing.PrefixTreeRouting`.

    For each (model, prompt) it generates a response, grades it to a quality in [0, 1], and
    accumulates token cost. Returns the ``(n_models, n_prompts)`` quality matrix and the
    per-model average cost per query (USD). Pricing comes from ``client.price_per_1k`` unless
    a ``pricing`` override dict provides an entry for the model. ``len(prompts)`` should equal
    ``branching ** depth`` for the routing tree.
    """
    n_m, n_p = len(model_ids), len(prompts)
    quality = np.zeros((n_m, n_p), dtype=np.float64)
    costs = np.zeros(n_m, dtype=np.float64)
    for mi, model_id in enumerate(model_ids):
        override = pricing.get(model_id) if pricing else None
        in_price, out_price = override if override is not None else client.price_per_1k(model_id)
        total = 0.0
        for pi, prompt in enumerate(prompts):
            text, in_tok, out_tok = client.generate(model_id, prompt)
            quality[mi, pi] = float(np.clip(grade(prompt, text), 0.0, 1.0))
            total += in_price * in_tok / 1000.0 + out_price * out_tok / 1000.0
        costs[mi] = total / max(1, n_p)
    return quality, costs


class MockLLMClient:
    """Deterministic in-memory :class:`LLMClient` for tests and offline demos.

    ``responder(model_id, prompt) -> text`` produces the completion (defaults to a canned
    echo). Token counts are word counts, so cost accounting is exercised without any network
    or SDK. ``pricing`` maps ``model_id -> (in_price, out_price)`` per 1K tokens.
    """

    def __init__(
        self,
        responder: Callable[[str, str], str] | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        default_price: tuple[float, float] = (0.001, 0.001),
    ) -> None:
        self._responder = responder or (lambda model_id, prompt: f"[{model_id}] re: {prompt[:24]}")
        self.pricing = pricing or {}
        self.default_price = default_price

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        text = self._responder(model_id, prompt)
        if max_tokens is not None:
            text = " ".join(text.split()[:max_tokens])
        return text, len(prompt.split()), len(text.split())

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        return self.pricing.get(model_id, self.default_price)
