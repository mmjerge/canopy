"""Prefix-cache optimization as online selection over a token trie.

LLM serving systems (vLLM automatic prefix caching, SGLang RadixAttention) cache the KV
state of common token prefixes: many prompts share a prefix, so it is computed once and
reused. The cache is an ancestor-closed subtree of the token trie under a memory budget,
and the value of caching a prefix aggregates over all prompts that pass through it -- the
same tree-structured, aggregate-feedback, storage-bounded problem as the rest of this
package, with **storage = cache memory**.

We model a stream of prompts (token sequences), where caching prefix node ``v`` saves one
token of recompute for every prompt that passes through it. Savings on a prompt = the
depth of the deepest cached prefix on its path. The policies maintain an ancestor-closed
resident set of at most ``budget`` nodes (evicting only cached *leaves*, as a radix cache
does) and differ only in the per-node score used for admission/eviction:

* ``lru``      -- least-recently-used (recency).
* ``lfu``      -- least-frequently-used (raw count).
* ``adaptive`` -- recency-weighted frequency (EWMA of hits): tracks a drifting prompt
  distribution faster than LFU while staying frequency-aware unlike LRU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Prompt = tuple[int, ...]


class PrefixCacheEnv:
    """A (optionally non-stationary) stream of prompts sharing cacheable prefixes.

    Each prompt is a popular ``prefix_len``-token template (the cacheable shared part)
    followed by a random ``suffix_len``-token tail. Template popularity is Zipfian and, if
    ``shift_at`` is set, is permuted at that round to model a distribution shift.
    """

    def __init__(
        self,
        vocab: int = 8,
        prefix_len: int = 4,
        suffix_len: int = 3,
        n_templates: int = 40,
        zipf_s: float = 1.2,
        shift_at: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.vocab = vocab
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.n_templates = n_templates
        self.shift_at = shift_at
        self.rng = rng or np.random.default_rng()
        # distinct template prefixes
        self.templates = [tuple(int(t) for t in self.rng.integers(0, vocab, prefix_len))
                          for _ in range(n_templates)]
        ranks = np.arange(1, n_templates + 1)
        w = 1.0 / ranks**zipf_s
        self.base_pop = w / w.sum()
        self.shift_pop = self.base_pop[self.rng.permutation(n_templates)]

    def popularity(self, t: int) -> NDArray[np.float64]:
        if self.shift_at is not None and t >= self.shift_at:
            return self.shift_pop
        return self.base_pop

    def next_prompt(self, t: int) -> Prompt:
        k = int(self.rng.choice(self.n_templates, p=self.popularity(t)))
        suffix = tuple(int(s) for s in self.rng.integers(0, self.vocab, self.suffix_len))
        return self.templates[k] + suffix

    def generate_stream(self, horizon: int) -> list[Prompt]:
        """Realize a prompt stream of length ``horizon`` (encodes any popularity shift)."""
        return [self.next_prompt(t) for t in range(horizon)]


@dataclass
class CacheResult:
    avg_savings: float  # mean tokens reused per prompt (last window)
    savings_curve: NDArray[np.float64]  # rolling-mean savings over the stream
    label: str


def _savings_curve(prompts, resident_at, window: int) -> tuple[NDArray[np.float64], float]:
    """Helper: rolling-mean savings given a callable resident_at(t) -> set (or fixed set)."""
    n = len(prompts)
    savings = np.empty(n)
    recent: list[int] = []
    run_sum = 0.0
    for t, prompt in enumerate(prompts):
        resident = resident_at(t)
        saved = 0
        for i in range(1, len(prompt) + 1):
            if prompt[:i] in resident:
                saved = i
        recent.append(saved)
        if len(recent) > window:
            run_sum -= recent.pop(0)
        run_sum += saved
        savings[t] = run_sum / len(recent)
    return savings, float(savings[-window:].mean())


def offline_optimal(prompts, budget: int, window: int = 500) -> CacheResult:
    """Best static budget-B cache in hindsight: top-B prefix nodes by true frequency.

    Node frequency is monotone non-increasing with depth (a child is hit no more often
    than its parent), so the top-B nodes by count are automatically ancestor-closed --
    hence an optimal static cache for the 1-token-per-node savings model.
    """
    counts: dict[Prompt, int] = {}
    for prompt in prompts:
        for i in range(1, len(prompt) + 1):
            nd = prompt[:i]
            counts[nd] = counts.get(nd, 0) + 1
    resident = set(sorted(counts, key=counts.get, reverse=True)[:budget])  # type: ignore[arg-type]
    curve, avg = _savings_curve(prompts, lambda _t: resident, window)
    return CacheResult(avg_savings=avg, savings_curve=curve, label="offline")


def run_cache(
    prompts,
    budget: int,
    policy: str = "adaptive",
    decay: float = 0.995,
    window: int = 500,
) -> CacheResult:
    """Run an online prefix-cache policy and measure tokens reused per prompt.

    ``policy`` is one of ``lru`` (recency), ``lfu`` (count), ``adaptive`` (recency-weighted
    frequency / EWMA). The resident set stays ancestor-closed and only cached leaves are
    evicted, as a radix KV cache does.
    """
    if policy == "offline":
        return offline_optimal(prompts, budget, window)

    resident: set[Prompt] = set()
    cached_children: dict[Prompt, int] = {}
    count: dict[Prompt, float] = {}
    last: dict[Prompt, int] = {}
    ewma: dict[Prompt, float] = {}

    def score(nd: Prompt) -> float:
        if policy == "lru":
            return float(last.get(nd, -1))
        if policy == "lfu":
            return count.get(nd, 0.0)
        return ewma.get(nd, 0.0)  # adaptive

    savings = np.empty(len(prompts))
    recent: list[int] = []
    run_sum = 0.0
    for t, prompt in enumerate(prompts):
        path = [prompt[:i] for i in range(1, len(prompt) + 1)]

        saved = 0
        for i, nd in enumerate(path, start=1):
            if nd in resident:
                saved = i
        recent.append(saved)
        if len(recent) > window:
            run_sum -= recent.pop(0)
        run_sum += saved
        savings[t] = run_sum / len(recent)

        for nd in path:
            count[nd] = count.get(nd, 0.0) + 1.0
            ewma[nd] = ewma.get(nd, 0.0) * decay ** (t - last.get(nd, t)) + 1.0
            last[nd] = t

        for nd in path:
            if nd in resident:
                continue
            parent = nd[:-1]
            if len(nd) > 1 and parent not in resident:
                break
            if len(resident) < budget:
                resident.add(nd)
                cached_children[nd] = 0
                if len(nd) > 1:
                    cached_children[parent] = cached_children.get(parent, 0) + 1
            else:
                leaves = [r for r in resident if cached_children.get(r, 0) == 0]
                victim = min(leaves, key=score)
                if score(nd) > score(victim):
                    resident.discard(victim)
                    vp = victim[:-1]
                    if len(victim) > 1:
                        cached_children[vp] = cached_children.get(vp, 1) - 1
                    cached_children.pop(victim, None)
                    resident.add(nd)
                    cached_children[nd] = 0
                    if len(nd) > 1:
                        cached_children[parent] = cached_children.get(parent, 0) + 1
                else:
                    break

    return CacheResult(avg_savings=float(savings[-window:].mean()),
                       savings_curve=savings, label=policy)
