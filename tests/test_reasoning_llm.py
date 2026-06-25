"""Tests for the real-LLM reasoning-search harness, using a deterministic mock generator."""

from __future__ import annotations

import numpy as np

from oco.bandits.reasoning_llm import (
    best_of_n,
    extract_answer,
    is_correct,
    value_guided_search,
)


def test_answer_extraction_and_grading():
    assert extract_answer("blah #### 42") == "42"
    assert extract_answer("the cost is $1,200.00 total") == "1200"
    assert extract_answer("no numbers here") is None
    assert is_correct("... so #### 18", "18")
    assert not is_correct("... so #### 17", "18")


class MockLLM:
    """A toy 'reasoning' model with K binary decisions and a GRADED answer.

    The partial solution encodes the decisions made so far as bits (1 = correct). A 'next
    step' appends one bit, correct with probability ``p_step``. A completion fills the
    remaining decisions and returns the count of correct decisions as the answer (graded,
    PRM-style partial credit); the fully-correct answer is ``K``. So a graded value (mean
    rollout score) is informative at every step, while best-of-N's majority vote concentrates
    near ``K * p_step`` and almost never returns ``K``.
    """

    def __init__(self, k: int, p_step: float = 0.5):
        self.k = k
        self.p_step = p_step

    @staticmethod
    def _bits(prefix: str) -> str:
        return "".join(ch for ch in prefix if ch in "01")

    def __call__(self, prompt: str, max_tokens: int, seed: int) -> str:
        rng = np.random.default_rng(seed)
        prefix = prompt.split("Solution so far:")[-1] if "Solution so far:" in prompt else ""
        made = list(self._bits(prefix))
        if "Next step:" in prompt:
            return f"step -> {'1' if rng.random() < self.p_step else '0'}"
        bits = made + ["1" if rng.random() < self.p_step else "0"
                       for _ in range(self.k - len(made))]
        return f"reasoning... #### {sum(b == '1' for b in bits)}"


def _success(method, k, seeds=150, **kw):
    out = []
    for s in range(seeds):
        gen = MockLLM(k)

        def g(prompt, max_tokens, seed, _gen=gen, _s=s):
            return _gen(prompt, max_tokens, seed + 7919 * _s)

        out.append(method("q", gold=str(k), generate=g, **kw).correct)
    return float(np.mean(out))


def test_methods_run_and_count_budget():
    gen = MockLLM(k=4)
    r = best_of_n("q", gold="4", generate=gen, n=5)
    assert r.budget.calls == 5 and r.answer is not None
    r2 = value_guided_search("q", gold="4", generate=MockLLM(4),
                             branching=2, n_steps=4, rollouts=2)
    # branching*(1 + rollouts) per step, plus one final rollout
    assert r2.budget.calls == 4 * (2 * (1 + 2)) + 1


def test_value_guided_beats_best_of_n_given_informative_value():
    # graded (PRM-style) value = mean rollout score -> the search steers to the fully-correct
    # trace, where best-of-N's majority vote concentrates near K*p_step and misses K.
    def graded_value(rolls):
        scores = [extract_answer(t) for t in rolls]
        return float(np.mean([float(s) for s in scores if s is not None])) if scores else 0.0

    k = 5
    vg = _success(value_guided_search, k, branching=3, n_steps=k, rollouts=3,
                  value_fn=graded_value)
    bo = _success(best_of_n, k, n=3 * k * 4 + 1)  # comparable call budget
    assert vg > 0.3
    assert vg > bo + 0.2
