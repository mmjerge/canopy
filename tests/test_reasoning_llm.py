"""Tests for the real-LLM reasoning-search harness, using a deterministic mock generator."""

from __future__ import annotations

import numpy as np

from canopy.bandits.reasoning_llm import (
    best_of_n,
    extract_answer,
    extract_boxed,
    extract_letter,
    is_correct,
    value_guided_search,
)


def test_boxed_extraction():
    assert extract_boxed(r"... so the answer is \boxed{42}.") == "42"
    assert extract_boxed(r"nested \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    # last boxed wins; falls back to numeric extraction when no \boxed present
    assert extract_boxed(r"\boxed{1} then finally \boxed{7}") == "7"
    assert extract_boxed("no box, answer is 12") == "12"
    assert extract_boxed("nothing here") is None


def test_letter_extraction():
    assert extract_letter("I think the answer is B.") == "B"
    assert extract_letter("Answer: C") == "C"
    # standalone letters only: 'A' inside a word must not match
    assert extract_letter("Available data suggests... Answer: D") == "D"
    assert extract_letter("no letters 123") is None


def test_instrumented_search_logs_cheap_and_true():
    """The instrumented descent logs per-candidate cheap vs true values and follows
    the cheap argmax; with an informative generator the pivotal hit is recorded."""
    from canopy.bandits.reasoning_llm import instrumented_value_guided_search

    # generator: candidate c=0 leads to correct rollouts ('#### 7'), c>0 to wrong ones.
    # rollout seeds are 50_000 + 1000*(step*branching + c) + r, so c = (seed-50_000)//1000 % 3.
    def gen(prompt, max_tokens, seed):
        if "Next step:" in prompt:
            return f"step{seed % 1000}"
        if 50_000 <= seed < 999_000:
            c = ((seed - 50_000) // 1000) % 3
            return "#### 7" if c == 0 else "#### 1"
        return "#### 7"  # final rollouts from the (correct) chosen prefix

    res, logs = instrumented_value_guided_search(
        "q", gold="7", generate=gen, branching=3, n_steps=2, rollouts=2
    )
    assert len(logs) == 2
    for lg in logs:
        assert len(lg.cheap) == 3 and len(lg.true) == 3
        assert lg.chosen == int(np.argmax(lg.cheap))  # follows the cheap edge
        assert 0.0 <= lg.spread <= 1.0 and lg.gap >= 0.0
    # candidate 0's rollouts grade correct -> true value 1.0, others 0.0 -> pivotal
    assert logs[0].true[0] == 1.0 and max(logs[0].true[1:]) == 0.0
    assert logs[0].spread == 1.0


def test_extract_fn_plumbs_through_search():
    """best_of_n / value_guided_search grade with the supplied extractor."""

    def gen(prompt, max_tokens, seed):
        return r"reasoning... \boxed{9}"

    r = best_of_n("q", gold="9", generate=gen, n=3, extract=extract_boxed)
    assert r.correct and r.answer == "9"
    r2 = value_guided_search(
        "q", gold="9", generate=gen, branching=2, n_steps=2, rollouts=1, extract=extract_boxed
    )
    assert r2.correct and r2.answer == "9"


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
        bits = made + [
            "1" if rng.random() < self.p_step else "0" for _ in range(self.k - len(made))
        ]
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
    r2 = value_guided_search("q", gold="4", generate=MockLLM(4), branching=2, n_steps=4, rollouts=2)
    # branching*(1 + rollouts) per step, plus one final rollout
    assert r2.budget.calls == 4 * (2 * (1 + 2)) + 1


def test_value_guided_beats_best_of_n_given_informative_value():
    # graded (PRM-style) value = mean rollout score -> the search steers to the fully-correct
    # trace, where best-of-N's majority vote concentrates near K*p_step and misses K.
    def graded_value(rolls):
        scores = [extract_answer(t) for t in rolls]
        return float(np.mean([float(s) for s in scores if s is not None])) if scores else 0.0

    k = 5
    vg = _success(value_guided_search, k, branching=3, n_steps=k, rollouts=3, value_fn=graded_value)
    bo = _success(best_of_n, k, n=3 * k * 4 + 1)  # comparable call budget
    assert vg > 0.3
    assert vg > bo + 0.2
