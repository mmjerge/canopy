"""Tests for the caching + budget-capped LLM client wrapper."""

from __future__ import annotations

import pytest

from canopy.llm import BudgetError, CachingLLMClient, LLMClient, MockLLMClient


class CountingClient(MockLLMClient):
    """Mock client that counts how many times the underlying generate is hit."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_calls = 0

    def generate(self, model_id, prompt, temperature=None, max_tokens=None):
        self.n_calls += 1
        return super().generate(model_id, prompt, temperature, max_tokens)


def test_cache_hits_avoid_repeat_calls(tmp_path):
    inner = CountingClient(pricing={"m": (1.0, 1.0)})
    client = CachingLLMClient(inner, tmp_path / "cache.jsonl")
    a = client.generate("m", "hello world")
    b = client.generate("m", "hello world")  # identical -> served from memory cache
    assert a == b
    assert inner.n_calls == 1
    assert client.stats()["cache_hits"] == 1
    assert client.stats()["calls"] == 1


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "cache.jsonl"
    inner = CountingClient()
    CachingLLMClient(inner, path).generate("m", "p")
    assert inner.n_calls == 1
    # a fresh wrapper over a fresh inner client should serve from disk, no new call
    inner2 = CountingClient()
    out = CachingLLMClient(inner2, path).generate("m", "p")
    assert inner2.n_calls == 0
    assert out[0]  # non-empty text from disk


def test_satisfies_protocol(tmp_path):
    assert isinstance(CachingLLMClient(MockLLMClient(), tmp_path / "c.jsonl"), LLMClient)


def test_call_cap_blocks_before_issuing(tmp_path):
    inner = CountingClient()
    client = CachingLLMClient(inner, tmp_path / "c.jsonl", max_calls=2)
    client.generate("m", "a")
    client.generate("m", "b")
    with pytest.raises(BudgetError):
        client.generate("m", "c")  # third distinct prompt exceeds the cap
    assert inner.n_calls == 2  # the blocked call was never issued


def test_spend_cap_trips(tmp_path):
    # each call costs len(prompt.split()) in/out tokens * price; make it exceed quickly
    inner = CountingClient(pricing={"m": (1000.0, 1000.0)})
    client = CachingLLMClient(inner, tmp_path / "c.jsonl", max_spend_usd=0.001)
    with pytest.raises(BudgetError):
        client.generate("m", "one two three four five")
    assert client.spend > 0.001
