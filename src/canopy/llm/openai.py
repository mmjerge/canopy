"""OpenAI implementation of :class:`~canopy.llm.base.LLMClient`.

Uses the Chat Completions API. Requires the optional ``openai`` extra
(``uv sync --extra openai``) and an ``OPENAI_API_KEY`` in the environment (or passed
explicitly). Set ``base_url`` to target an OpenAI-compatible endpoint (Azure OpenAI,
local servers, etc.).
"""

from __future__ import annotations

import os

from canopy.llm.base import Generation

# Approximate USD price per 1K tokens (input, output); update as pricing changes.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o3-mini": (0.0011, 0.0044),
}


class OpenAIClient:
    """Wrapper over the OpenAI Chat Completions API implementing :class:`LLMClient`.

    Args:
        api_key: API key; defaults to the ``OPENAI_API_KEY`` environment variable.
        base_url: Optional override for an OpenAI-compatible endpoint.
        max_tokens: Default generation cap.
        pricing: Optional ``model_id -> (in, out)`` per-1K-token price table.
        client: Optional pre-built OpenAI SDK client (for injection/testing).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 512,
        pricing: dict[str, tuple[float, float]] | None = None,
        client: object | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.pricing = pricing or DEFAULT_PRICING
        if client is not None:
            self._client = client
            return
        from openai import OpenAI  # lazy: only needed when actually calling OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generation:
        """Return ``(response_text, input_tokens, output_tokens)`` for one prompt."""
        kwargs: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, int(usage.prompt_tokens), int(usage.completion_tokens)

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        """Per-1K-token ``(input, output)`` USD price for ``model_id`` (0 if unknown)."""
        return self.pricing.get(model_id, (0.0, 0.0))
