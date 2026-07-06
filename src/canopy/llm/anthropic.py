"""Anthropic (Claude) implementation of :class:`~canopy.llm.base.LLMClient`.

Uses the Messages API. Requires the optional ``anthropic`` extra
(``uv sync --extra anthropic``) and an ``ANTHROPIC_API_KEY`` in the environment (or passed
explicitly). Set ``base_url`` to target a compatible endpoint (e.g. a proxy).
"""

from __future__ import annotations

import os
from typing import Any

from canopy.llm.base import Generation

# Approximate USD price per 1K tokens (input, output); update as pricing changes.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-latest": (0.003, 0.015),
    "claude-3-5-haiku-latest": (0.0008, 0.004),
    "claude-3-opus-latest": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}


class AnthropicClient:
    """Wrapper over the Anthropic Messages API implementing :class:`LLMClient`.

    Args:
        api_key: API key; defaults to the ``ANTHROPIC_API_KEY`` environment variable.
        base_url: Optional override for a compatible endpoint.
        max_tokens: Default generation cap (the Messages API requires ``max_tokens``).
        pricing: Optional ``model_id -> (in, out)`` per-1K-token price table.
        client: Optional pre-built Anthropic SDK client (for injection/testing).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 512,
        pricing: dict[str, tuple[float, float]] | None = None,
        client: Any | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.pricing = pricing or DEFAULT_PRICING
        if client is not None:
            self._client = client
            return
        from anthropic import Anthropic  # lazy: only needed when actually calling Anthropic

        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), base_url=base_url
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        """Return ``(response_text, input_tokens, output_tokens)`` for one prompt.

        ``seed`` distinguishes independent same-prompt samples for caching (the API samples at
        ``temperature``); it is not forwarded to the model.
        """
        kwargs: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        usage = resp.usage
        return text, int(usage.input_tokens), int(usage.output_tokens)

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        """Per-1K-token ``(input, output)`` USD price for ``model_id`` (0 if unknown)."""
        return self.pricing.get(model_id, (0.0, 0.0))
