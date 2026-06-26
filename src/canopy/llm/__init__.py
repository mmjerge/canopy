"""Provider-agnostic LLM clients for the applied (routing / reasoning) experiments.

The :class:`LLMClient` Protocol is the contract; concrete providers implement it and are
imported lazily (their SDKs are optional extras). Everything downstream depends on the
Protocol, so swapping providers needs no algorithm changes.

    from canopy.llm import BedrockClient, OpenAIClient, AnthropicClient, measure_quality_matrix

``DEFAULT_PRICING`` re-exports the Bedrock price table for backward compatibility.
"""

from canopy.llm.anthropic import AnthropicClient
from canopy.llm.base import (
    GenerateFn,
    Generation,
    LLMClient,
    MockLLMClient,
    as_generate_fn,
    measure_quality_matrix,
)
from canopy.llm.bedrock import DEFAULT_PRICING, BedrockClient
from canopy.llm.openai import OpenAIClient

__all__ = [
    "LLMClient",
    "Generation",
    "GenerateFn",
    "as_generate_fn",
    "measure_quality_matrix",
    "MockLLMClient",
    "BedrockClient",
    "OpenAIClient",
    "AnthropicClient",
    "DEFAULT_PRICING",
]
