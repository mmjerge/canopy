"""Tests for the provider-agnostic LLM client layer (canopy.llm)."""

from __future__ import annotations

import types

import numpy as np

from canopy.llm import (
    AnthropicClient,
    BedrockClient,
    LLMClient,
    MockLLMClient,
    OpenAIClient,
    as_generate_fn,
    measure_quality_matrix,
)


def test_mock_client_generate_and_pricing():
    client = MockLLMClient(
        responder=lambda model, prompt: "one two three",
        pricing={"m": (1.0, 2.0)},
    )
    text, in_tok, out_tok = client.generate("m", "a b")
    assert text == "one two three"
    assert (in_tok, out_tok) == (2, 3)
    assert client.price_per_1k("m") == (1.0, 2.0)
    assert client.price_per_1k("unknown") == (0.001, 0.001)  # default


def test_mock_client_respects_max_tokens():
    client = MockLLMClient(responder=lambda model, prompt: "a b c d e")
    text, _, out_tok = client.generate("m", "x", max_tokens=2)
    assert text == "a b"
    assert out_tok == 2


def test_clients_satisfy_protocol():
    # runtime_checkable Protocol: all concrete clients are LLMClients.
    assert isinstance(MockLLMClient(), LLMClient)
    assert isinstance(BedrockClient(runtime=object()), LLMClient)
    assert isinstance(OpenAIClient(client=object()), LLMClient)
    assert isinstance(AnthropicClient(client=object()), LLMClient)


def test_measure_quality_matrix_shapes_and_cost():
    prompts = ["p0", "p1", "p2", "p3"]
    models = ["big", "small"]
    # grade: "big" always 1.0, "small" always 0.5 (grade sees prompt + text).
    client = MockLLMClient(
        responder=lambda model, prompt: f"resp {model}",
        pricing={"big": (10.0, 10.0), "small": (1.0, 1.0)},
    )

    def grade(prompt: str, text: str) -> float:
        return 1.0 if "big" in text else 0.5

    quality, costs = measure_quality_matrix(prompts, models, client, grade)
    assert quality.shape == (2, 4)
    assert np.allclose(quality[0], 1.0)
    assert np.allclose(quality[1], 0.5)
    # big is more expensive than small (same tokens, 10x price)
    assert costs[0] > costs[1] > 0


def test_measure_quality_matrix_pricing_override():
    client = MockLLMClient(responder=lambda model, prompt: "x", pricing={"m": (1.0, 1.0)})
    _, costs_default = measure_quality_matrix(["a"], ["m"], client, lambda p, t: 1.0)
    _, costs_override = measure_quality_matrix(
        ["a"], ["m"], client, lambda p, t: 1.0, pricing={"m": (100.0, 100.0)}
    )
    assert costs_override[0] > costs_default[0]


def test_as_generate_fn_adapts_to_reasoning_signature():
    seen = {}

    def responder(model, prompt):
        seen["model"] = model
        return "step"

    client = MockLLMClient(responder=responder)
    generate = as_generate_fn(client, "my-model", temperature=0.5)
    out = generate("prompt", 16, 7)  # (prompt, max_tokens, seed)
    assert out == "step"
    assert seen["model"] == "my-model"


def _fake_openai_client(text: str, prompt_tokens: int, completion_tokens: int):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    response = types.SimpleNamespace(choices=[choice], usage=usage)

    create_calls = {}

    def create(**kwargs):
        create_calls.update(kwargs)
        return response

    completions = types.SimpleNamespace(create=create)
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(chat=chat), create_calls


def test_openai_client_parses_response():
    fake, calls = _fake_openai_client("hello world", 11, 5)
    client = OpenAIClient(client=fake, pricing={"gpt-x": (1.0, 2.0)})
    text, in_tok, out_tok = client.generate("gpt-x", "hi", temperature=0.3, max_tokens=64)
    assert (text, in_tok, out_tok) == ("hello world", 11, 5)
    assert calls["model"] == "gpt-x"
    assert calls["messages"] == [{"role": "user", "content": "hi"}]
    assert calls["max_tokens"] == 64
    assert calls["temperature"] == 0.3
    assert client.price_per_1k("gpt-x") == (1.0, 2.0)


def test_bedrock_client_parses_response():
    class FakeRuntime:
        def __init__(self):
            self.last = None

        def converse(self, **kwargs):
            self.last = kwargs
            return {
                "output": {"message": {"content": [{"text": "answer"}]}},
                "usage": {"inputTokens": 12, "outputTokens": 7},
            }

    runtime = FakeRuntime()
    client = BedrockClient(runtime=runtime)
    text, in_tok, out_tok = client.generate("model-x", "q", temperature=0.7, max_tokens=128)
    assert (text, in_tok, out_tok) == ("answer", 12, 7)
    assert runtime.last["modelId"] == "model-x"
    assert runtime.last["inferenceConfig"]["maxTokens"] == 128
    assert runtime.last["inferenceConfig"]["temperature"] == 0.7


def test_anthropic_client_parses_response():
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        block = types.SimpleNamespace(text="claude says hi")
        usage = types.SimpleNamespace(input_tokens=9, output_tokens=4)
        return types.SimpleNamespace(content=[block], usage=usage)

    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    client = AnthropicClient(client=fake, pricing={"claude-x": (3.0, 15.0)})
    text, in_tok, out_tok = client.generate("claude-x", "hi", temperature=0.2, max_tokens=32)
    assert (text, in_tok, out_tok) == ("claude says hi", 9, 4)
    assert calls["model"] == "claude-x"
    assert calls["messages"] == [{"role": "user", "content": "hi"}]
    assert calls["max_tokens"] == 32
    assert calls["temperature"] == 0.2
    assert client.price_per_1k("claude-x") == (3.0, 15.0)
