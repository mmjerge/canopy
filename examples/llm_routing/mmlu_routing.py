"""Real MMLU-by-subject routing benchmark over diverse Bedrock models.

Each MMLU subject is a prefix region of the routing tree, so the router can learn which
model to use per subject. Models span providers and tiers (Llama 70B/8B, Nova Pro/Lite/
Micro, Mistral-Small) so they have genuinely complementary strengths -- the condition the
earlier toy benchmark lacked.

Run:  uv run --extra bench --extra plot python examples/llm_routing/mmlu_routing.py
Use --all-models to discover and route over every invokable Bedrock text model in the region,
or --models a,b,c to pass an explicit list. Needs AWS creds with Bedrock access. Caches
per-model results to mmlu_routing.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canopy.bandits import PrefixTreeRouting, run_router
from canopy.llm import DEFAULT_PRICING, BedrockClient, BudgetError, CachingLLMClient

SUBJECTS = [
    "elementary_mathematics",
    "abstract_algebra",
    "high_school_biology",
    "college_computer_science",
    "world_religions",
    "moral_scenarios",
    "global_facts",
    "high_school_government_and_politics",
]
Q_PER = 4  # questions per subject -> 8 subjects * 4 = 32 leaves (branching=2, depth=5)
# Curated complementary set spanning a ~100x cost gradient (cheap -> frontier), chosen from a
# full --all-models sweep so that no single model dominates after cost: regional routing then
# genuinely beats any fixed model. Override with --models or widen with --all-models.
MODELS = [
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "qwen.qwen3-coder-next",
    "us.anthropic.claude-opus-4-7",
    "us.writer.palmyra-x5-v1:0",
]
LETTERS = "ABCD"

# Models whose IDs match these hints reason before answering and need a larger token budget;
# everything else is a multiple-choice classifier that only needs to emit a single letter.
REASONING_HINTS = ("deepseek", "-r1", ".r1", "reasoning", "qwq", "magistral")


def max_tokens_for(model_id: str, answer_tokens: int, reasoning_tokens: int) -> int:
    mid = model_id.lower()
    return reasoning_tokens if any(h in mid for h in REASONING_HINTS) else answer_tokens


def load_questions() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    prompts, answers = [], []
    for subject in SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds.select(range(Q_PER)):
            opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(row["choices"]))
            prompts.append(
                f"Question: {row['question']}\n{opts}\n"
                "Answer with only the letter A, B, C, or D."
            )
            answers.append(LETTERS[int(row["answer"])])
    return prompts, answers


def parse_letter(text: str) -> str:
    # Scan from the end: reasoning models mention options mid-thought, but the final answer
    # is last; terse classifiers emit a single letter, so either direction agrees.
    for ch in reversed(text.upper()):
        if ch in LETTERS:
            return ch
    return "?"


def _audit(region: str) -> None:
    """Print every discovered model's availability status, grouped, then exit."""
    base = BedrockClient(region=region, max_tokens=8)
    ids = base.list_text_models(only_available=False)
    print(f"{len(ids)} candidate text models in {region}\n")
    buckets: dict[str, list[str]] = {}
    for mid in ids:
        av = base.model_availability(mid.split("/")[-1])
        if av is None:
            key = "unknown (no availability info / inference profile)"
        elif (
            av["authorizationStatus"] == "AUTHORIZED"
            and av["regionAvailability"] == "AVAILABLE"
            and av["entitlementAvailability"] == "AVAILABLE"
        ):
            key = "INVOKABLE"
        else:
            key = (
                f"auth={av['authorizationStatus']} "
                f"entitled={av['entitlementAvailability']} "
                f"region={av['regionAvailability']}"
            )
        buckets.setdefault(key, []).append(mid)
    for key in sorted(buckets):
        print(f"[{key}]  ({len(buckets[key])})")
        for mid in buckets[key]:
            print(f"    {mid}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--max-calls", type=int, default=None, help="hard cap on API calls")
    ap.add_argument("--max-spend", type=float, default=None, help="hard cap on est. USD spend")
    ap.add_argument("--cache", default="examples/.cache/mmlu_routing.jsonl")
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="discover and use every invokable Bedrock text model in the region",
    )
    ap.add_argument(
        "--models",
        default="",
        help="comma-separated model IDs to use (overrides the default list and --all-models)",
    )
    ap.add_argument(
        "--answer-tokens",
        type=int,
        default=8,
        help="generation cap for plain multiple-choice models (just emits a letter)",
    )
    ap.add_argument(
        "--reasoning-tokens",
        type=int,
        default=1024,
        help="generation cap for reasoning models (DeepSeek R1 etc.) that think first",
    )
    ap.add_argument(
        "--include-unavailable",
        action="store_true",
        help="with --all-models, skip the availability pre-filter and attempt every catalog ID",
    )
    ap.add_argument(
        "--audit-models",
        action="store_true",
        help="print each discovered model's authorization/entitlement/region status and exit",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore the .npz aggregate and re-measure (individual calls still hit the JSONL "
        "response cache, so already-seen model/prompt pairs stay free)",
    )
    args = ap.parse_args()

    if args.audit_models:
        _audit(args.region)
        return

    npz_cache = Path(__file__).parent / "mmlu_routing.npz"
    prompts, answers = load_questions()
    n = len(prompts)

    # Determine the target model set.
    if args.models.strip():
        candidates = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.all_models:
        base0 = BedrockClient(region=args.region, max_tokens=8)
        candidates = base0.list_text_models(only_available=not args.include_unavailable)
        tag = "all catalog" if args.include_unavailable else "available"
        print(f"discovered {len(candidates)} {tag} text models in {args.region}")
    else:
        candidates = list(MODELS)
    if not candidates:
        print("No candidate models found. Check credentials / region / model access.")
        return

    # Reuse any measurements already in the .npz cache; only query models not yet measured.
    cached: dict[str, tuple[np.ndarray, float]] = {}
    if npz_cache.exists() and not args.fresh:
        data = np.load(npz_cache, allow_pickle=True)
        if data["quality"].shape[1] == n:
            for i, m in enumerate(str(x) for x in data["models"]):
                cached[m] = (data["quality"][i], float(data["costs"][i]))
    missing = [m for m in candidates if m not in cached]
    if cached:
        print(
            f"reusing {len(candidates) - len(missing)}/{len(candidates)} models from "
            f"{npz_cache.name}; {len(missing)} to query"
        )

    if missing:
        client = CachingLLMClient(
            BedrockClient(region=args.region, max_tokens=8),
            args.cache,
            max_calls=args.max_calls,
            max_spend_usd=args.max_spend,
        )
        try:
            for model in missing:
                in_p, out_p = DEFAULT_PRICING.get(model, (0.001, 0.001))
                mt = max_tokens_for(model, args.answer_tokens, args.reasoning_tokens)
                row = np.zeros(n)
                cost = 0.0
                ok = True
                for pi, prompt in enumerate(prompts):
                    try:
                        text, it, ot = client.generate(model, prompt, max_tokens=mt)
                        row[pi] = 1.0 if parse_letter(text) == answers[pi] else 0.0
                        cost += in_p * it / 1000 + out_p * ot / 1000
                    except BudgetError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        # Drop a model that fails its first call (no access / not invokable).
                        if pi == 0:
                            print(f"  skip {model}: {type(e).__name__}: {str(e)[:100]}")
                            ok = False
                            break
                if not ok:
                    continue
                cached[model] = (row, cost / n)
                print(f"  {model:42s} acc={row.mean():.2f}  cost/q={cost / n:.6f}")
        except BudgetError as e:
            print(f"\n[budget stop] {e}  Saving everything measured so far.")
        # Persist the merged cache (it grows across runs).
        keys = list(cached.keys())
        np.savez(
            npz_cache,
            quality=np.array([cached[m][0] for m in keys]),
            costs=np.array([cached[m][1] for m in keys]),
            models=np.array(keys),
        )
        print(f"  budget: {client.stats()}")

    # Assemble the selected set in candidate order (skip any that could not be measured).
    models = [m for m in candidates if m in cached]
    if len(models) < 2:
        print(f"\nABORTING: only {len(models)} usable model(s); need at least 2 to route.")
        return
    quality = np.array([cached[m][0] for m in models])
    costs = np.array([cached[m][1] for m in models])
    print(f"routing over {len(models)} models")

    # per-subject accuracy: do models have complementary strengths?
    print("\nper-subject accuracy (rows=models):")
    print("  " + "  ".join(f"{s[:10]:>10s}" for s in SUBJECTS))
    for mi, model in enumerate(models):
        per_sub = [quality[mi, s * Q_PER : (s + 1) * Q_PER].mean() for s in range(len(SUBJECTS))]
        print(f"  {model.split('.')[-1][:14]:14s} " + " ".join(f"{v:10.2f}" for v in per_sub))

    rel_costs = costs / costs.max()
    print("\nrouting (branching=2, depth=5, region=subject at resolution 3):")
    print("  (only 'hierarchical' and 'flat' learn online; the rest are truth-based refs)")
    results = {}
    for strat, kw in [
        ("hierarchical", {"resolution": 3}),
        ("flat", {}),
        ("best_single", {}),
        ("all_largest", {}),
        ("oracle", {}),
    ]:
        env = PrefixTreeRouting(
            2, 5, quality, rel_costs, lam=0.3, noise_std=0.05, rng=np.random.default_rng(0)
        )
        r = run_router(env, 6000, np.random.default_rng(1), strategy=strat, **kw)
        results[r.label] = r
        print(
            f"  {r.label:18s} regret={r.final_regret:7.1f}  "
            f"avg_quality={r.avg_quality:.3f}  rel_cost/q={r.total_cost/6000:.3f}"
        )

    _write_tables(models, quality, costs, results)
    _plot(results, quality)


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _write_tables(models: list[str], quality: np.ndarray, costs: np.ndarray, results: dict) -> None:
    """Write LaTeX tables (per-model measurements + routing comparison) and echo to console."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import FIGURE_DIR as tdir

    tdir.mkdir(parents=True, exist_ok=True)
    acc = quality.mean(axis=1)
    order = np.argsort(acc)[::-1]

    # ---- per-model measurement table ----
    print("\nper-model results (sorted by accuracy):")
    print(f"  {'model':44s} {'acc':>5s} {'cost/q($)':>11s}")
    model_rows = []
    for i in order:
        print(f"  {models[i]:44s} {acc[i]:5.2f} {costs[i]:11.6f}")
        model_rows.append(
            f"\\texttt{{{_latex_escape(models[i])}}} & {acc[i]:.2f} & {costs[i]:.6f} \\\\"
        )
    model_tex = (
        "% MMLU per-model accuracy and cost (auto-generated by mmlu_routing.py)\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Model & Accuracy & Cost/query (\\$) \\\\\n\\midrule\n"
        + "\n".join(model_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (tdir / "mmlu_models_table.tex").write_text(model_tex)

    # ---- routing strategy comparison table ----
    pretty = {
        "hierarchical(r=3)": "Hierarchical (ours)",
        "flat": "Flat (structure-blind)",
        "best_single": "Best single (oracle ref.)",
        "all_largest": "Highest-quality (oracle ref.)",
        "oracle": "Per-prompt oracle",
    }
    horizon = len(next(iter(results.values())).cum_regret)
    strat_rows = []
    for key in ("hierarchical(r=3)", "flat", "best_single", "all_largest", "oracle"):
        if key not in results:
            continue
        r = results[key]
        strat_rows.append(
            f"{pretty.get(key, key)} & {r.final_regret:.0f} & {r.avg_quality:.3f} & "
            f"{r.total_cost / horizon:.3f} \\\\"
        )
    strat_tex = (
        "% MMLU routing comparison (auto-generated by mmlu_routing.py)\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Policy & Regret & Avg.\\ quality & Rel.\\ cost/query \\\\\n\\midrule\n"
        + "\n".join(strat_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (tdir / "mmlu_routing_table.tex").write_text(strat_tex)
    print(f"\nwrote LaTeX tables to {tdir}/mmlu_models_table.tex and mmlu_routing_table.tex")


def _plot(results: dict, quality: np.ndarray) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import matplotlib.pyplot as plt  # noqa: E402

    from _plotstyle import PALETTE, save_figure, set_style  # noqa: E402

    set_style()
    # The two online learners are solid; truth-based references are dashed/markers.
    styling = {
        "hierarchical(r=3)": (PALETTE["red"], "-", "o"),
        "flat": (PALETTE["blue"], "-", "s"),
        "best_single": (PALETTE["gray"], "--", "^"),
        "all_largest": (PALETTE["orange"], "--", "v"),
        "oracle": (PALETTE["green"], ":", "*"),
    }
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, r in results.items():
        color, ls, _ = styling.get(label, (PALETTE["purple"], "-", "o"))
        axA.plot(np.arange(1, len(r.cum_regret) + 1), r.cum_regret, color=color, ls=ls, label=label)
    axA.set_xlabel("prompts seen")
    axA.set_ylabel("cumulative routing regret")
    axA.set_title("Online learners (solid) vs. truth-based references (dashed)")
    axA.legend(loc="upper left")
    for label, r in results.items():
        color, _, marker = styling.get(label, (PALETTE["purple"], "-", "o"))
        axB.scatter(
            [r.total_cost / len(r.cum_regret)],
            [r.avg_quality],
            color=color,
            s=130,
            marker=marker,
            zorder=3,
        )
        axB.annotate(
            label,
            (r.total_cost / len(r.cum_regret), r.avg_quality),
            fontsize=8,
            xytext=(5, 4),
            textcoords="offset points",
        )
    axB.set_xlabel("relative cost per query")
    axB.set_ylabel("average quality (accuracy)")
    axB.set_title("Cost vs. quality")
    fig.suptitle(
        f"Routing over real MMLU subjects ({len(SUBJECTS)} subjects, "
        f"{quality.shape[0]} Bedrock models)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_figure(fig, "mmlu_routing")
    print(f"\nsaved chart to {out} (+ .png)")


if __name__ == "__main__":
    main()
