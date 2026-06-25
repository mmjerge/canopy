"""Matched-budget GSM8K diagnostic: best-of-N vs value-guided (self-consistency vs oracle value).

Three arms at a matched generation-call budget:
  * best-of-N (self-consistency majority vote);
  * value-guided search with the default self-consistency value (realistic test-time signal);
  * value-guided search with an ORACLE value (fraction of rollouts that are actually correct).

The third arm is a diagnostic, not a deployable method (it peeks at the gold answer to score
rollouts). Its purpose: if oracle-value search beats best-of-N but self-consistency search does
not, the bottleneck is the *value signal*, not the search machinery.
"""

from __future__ import annotations

import argparse

from canopy.bandits.reasoning_llm import best_of_n, extract_answer, value_guided_search


def load_gsm8k(n: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for row in ds.select(range(n)):
        out.append((row["question"], row["answer"].split("####")[-1].strip().replace(",", "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=12)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="us.meta.llama3-1-8b-instruct-v1:0")
    ap.add_argument("--branching", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--final-rollouts", type=int, default=5)
    args = ap.parse_args()

    from canopy.bandits.bedrock import BedrockClient
    client = BedrockClient(region=args.region, max_tokens=512)
    problems = load_gsm8k(args.n_problems)

    def generate(prompt: str, max_tokens: int, seed: int) -> str:
        text, _, _ = client.generate(args.model, prompt, temperature=0.7, max_tokens=max_tokens)
        return text

    # matched budget: value-guided's total calls = best-of-N's sample count
    vg_calls = args.n_steps * (args.branching * (1 + args.rollouts)) + args.final_rollouts

    tally = {"best_of_n": 0, "vg_self": 0, "vg_oracle": 0}
    for i, (q, gold) in enumerate(problems):
        def oracle_value(rolls, _gold=gold):
            from canopy.bandits.reasoning_llm import _normalize
            return sum(extract_answer(t) == _normalize(_gold) for t in rolls) / max(1, len(rolls))

        bo = best_of_n(q, gold, generate, n=vg_calls)
        vs = value_guided_search(q, gold, generate, args.branching, args.n_steps, args.rollouts,
                                 final_rollouts=args.final_rollouts)
        vo = value_guided_search(q, gold, generate, args.branching, args.n_steps, args.rollouts,
                                 value_fn=oracle_value, final_rollouts=args.final_rollouts)
        tally["best_of_n"] += bo.correct
        tally["vg_self"] += vs.correct
        tally["vg_oracle"] += vo.correct
        print(f"[{i+1}/{len(problems)}] gold={gold:>7}  bo={bo.answer}({int(bo.correct)})  "
              f"vg_self={vs.answer}({int(vs.correct)})  vg_oracle={vo.answer}({int(vo.correct)})",
              flush=True)

    n = len(problems)
    print(f"\nGSM8K {n} problems, matched budget ~{vg_calls} calls, model {args.model}")
    for name in ("best_of_n", "vg_self", "vg_oracle"):
        print(f"  {name:10s} acc {tally[name]/n:.2f}")


if __name__ == "__main__":
    main()
