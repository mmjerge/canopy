"""On-disk response caching and a spend cap for real-LLM runs.

:class:`CachingLLMClient` wraps any :class:`~canopy.llm.base.LLMClient` so that repeated runs
of an experiment are cheap (identical requests are served from disk) and bounded (a hard cap
on the number of API calls and/or estimated USD spend). This makes paper experiments
reproducible and safe to re-run: the first run pays for the calls, every subsequent run is
free, and a runaway loop is stopped by the budget guard.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from canopy.llm.base import Generation


class BudgetError(RuntimeError):
    """Raised when a configured call or spend cap would be exceeded."""


def _key(
    model_id: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int | None,
    seed: int | None = None,
) -> str:
    # ``seed`` is part of the key so independent same-prompt samples (self-consistency, the
    # branching candidates of value-guided search) are cached as distinct draws instead of
    # collapsing to the first completion. ``seed=None`` reproduces the old single-sample key.
    suffix = "" if seed is None else f"\x00{seed}"
    payload = f"{model_id}\x00{temperature}\x00{max_tokens}\x00{prompt}{suffix}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachingLLMClient:
    """An :class:`LLMClient` that caches responses on disk and enforces a budget.

    Args:
        client: the underlying provider client to wrap.
        cache_path: JSONL file storing ``key -> [text, in_tokens, out_tokens]`` entries.
        max_calls: hard cap on cache-missing API calls (``None`` for unlimited).
        max_spend_usd: hard cap on estimated cumulative USD spend (``None`` for unlimited).
    """

    def __init__(
        self,
        client: Any,
        cache_path: str | Path,
        max_calls: int | None = None,
        max_spend_usd: float | None = None,
    ) -> None:
        self._client = client
        self.cache_path = Path(cache_path)
        self.max_calls = max_calls
        self.max_spend_usd = max_spend_usd
        self.calls = 0
        self.hits = 0
        self.spend = 0.0
        self._lock = threading.Lock()
        self._cache: dict[str, list] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["k"]] = rec["v"]

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        key = _key(model_id, prompt, temperature, max_tokens, seed)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached[0], int(cached[1]), int(cached[2])
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetError(f"call cap reached ({self.max_calls}); not issuing more requests")
        # ``seed`` is used only to partition the cache: the wrapped client samples a fresh draw
        # at ``temperature`` on this cache miss, so distinct seeds yield distinct samples without
        # requiring the provider to honor a seed. (It is not forwarded, for provider-agnosticism.)
        text, in_tok, out_tok = self._client.generate(model_id, prompt, temperature, max_tokens)
        self.calls += 1
        in_price, out_price = self.price_per_1k(model_id)
        self.spend += in_price * in_tok / 1000.0 + out_price * out_tok / 1000.0
        self._store(key, [text, int(in_tok), int(out_tok)])
        if self.max_spend_usd is not None and self.spend > self.max_spend_usd:
            raise BudgetError(f"spend cap exceeded (${self.spend:.4f} > ${self.max_spend_usd})")
        return text, int(in_tok), int(out_tok)

    def _store(self, key: str, value: list) -> None:
        with self._lock:
            self._cache[key] = value
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"k": key, "v": value}) + "\n")

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        return self._client.price_per_1k(model_id)

    def stats(self) -> dict:
        """Return cache/budget counters for logging at the end of a run."""
        return {"calls": self.calls, "cache_hits": self.hits, "est_spend_usd": round(self.spend, 4)}
