"""Is the reasoning value function almost tree-K-Lipschitz? Measure it on real traces.

This is the direct measurement of the paper's prior (3.3) in the setting where it
lives most naturally. A node is a partial reasoning trace; its *true* value is the
probability a completion from it is correct (estimated by grading rollouts against
gold), and its *cheap* value is the self-consistency probe the search actually
follows. We instrument the value-guided descent to log both per node and test the two
halves of the prior:

  (i)  tree-Lipschitz backbone -- the cheap probe should track the true node value
       (cheap-vs-true correlation), so following the value edge is informative; and
  (ii) finitely many violations -- per problem, only a small number K of steps should
       be *pivotal* (large sibling true-value spread), where one choice is decisively
       better.

Outputs the Figure-11 panels (cheap-vs-true scatter; per-problem pivotal-step count
histogram) and the Table-8 statistics (Spearman/Pearson, edge-following hit rates
overall and on pivotal steps with a bootstrap CI, mean edge gap, pivotal fractions).

Run (needs AWS creds + Bedrock model access; makes real paid calls):
    uv run --extra llm --extra bench --extra plot \\
        python examples/analysis/reasoning_tree_lipschitz.py --benchmark math --n-problems 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reasoning"))

from canopy.bandits.reasoning_llm import instrumented_value_guided_search  # noqa: E402

THRESHOLDS = [0.25, 0.5, 0.75]
PIVOTAL_TAU = 0.5  # a step is pivotal when sibling true-value spread exceeds this


def characterize(items, generate, extract, templates, branching=3, n_steps=4, rollouts=3):
    """Run the instrumented search over ``items`` and pool the per-step logs."""
    _, cont_t, roll_t = templates
    all_logs, per_problem_k = [], []
    for i, (q, gold) in enumerate(items):
        _, logs = instrumented_value_guided_search(
            q,
            gold,
            generate,
            branching=branching,
            n_steps=n_steps,
            rollouts=rollouts,
            extract=extract,
            continue_template=cont_t,
            rollout_template=roll_t,
        )
        all_logs.extend(logs)
        per_problem_k.append(sum(1 for lg in logs if lg.spread > PIVOTAL_TAU))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} problems instrumented")
    return all_logs, np.array(per_problem_k)


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def statistics(all_logs, per_problem_k, n_boot: int = 5000, seed: int = 0):
    cheap = np.array([v for lg in all_logs for v in lg.cheap])
    true = np.array([v for lg in all_logs for v in lg.true])
    pearson = float(np.corrcoef(cheap, true)[0, 1])
    spearman = float(np.corrcoef(_rank(cheap), _rank(true))[0, 1])

    hits_all = np.array([lg.hit for lg in all_logs])
    pivotal = [lg for lg in all_logs if lg.spread > PIVOTAL_TAU]
    hits_piv = np.array([lg.hit for lg in pivotal]) if pivotal else np.array([])
    rng = np.random.default_rng(seed)
    if hits_piv.size:
        boots = np.array(
            [
                hits_piv[rng.integers(0, len(hits_piv), len(hits_piv))].mean()
                for _ in range(n_boot)
            ]
        )
        piv_ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    else:
        piv_ci = (float("nan"), float("nan"))
    gaps = np.array([lg.gap for lg in all_logs])
    fractions = {
        tau: float(np.mean([lg.spread > tau for lg in all_logs])) for tau in THRESHOLDS
    }
    return {
        "pearson": pearson,
        "spearman": spearman,
        "hit_all": float(hits_all.mean()),
        "hit_pivotal": float(hits_piv.mean()) if hits_piv.size else float("nan"),
        "hit_pivotal_ci": piv_ci,
        "n_pivotal": int(hits_piv.size),
        "mean_gap": float(gaps.mean()),
        "pivotal_fractions": fractions,
        "mean_k": float(per_problem_k.mean()),
        "cheap": cheap,
        "true": true,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="math", choices=["gsm8k", "math", "gpqa_diamond"])
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="us.meta.llama3-1-70b-instruct-v1:0")
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    args = ap.parse_args()

    try:
        from reasoning_search import load_benchmark  # shared loaders/templates

        from canopy.llm import BedrockClient, CachingLLMClient, as_generate_fn

        client = CachingLLMClient(
            BedrockClient(region=args.region, max_tokens=768),
            f"examples/.cache/reasoning_{args.benchmark}.jsonl",  # shares the sweep's cache
            max_calls=args.max_calls,
            max_spend_usd=args.max_spend,
        )
        items, extract, templates = load_benchmark(args.benchmark, args.n_problems)
        generate = as_generate_fn(client, args.model, temperature=0.7)
    except Exception as e:  # noqa: BLE001
        print(
            f"Could not initialize ({type(e).__name__}: {e}).\n"
            "Needs: uv sync --extra llm --extra bench; AWS creds with Bedrock access."
        )
        return

    print(f"{args.benchmark}: instrumenting {len(items)} problems, model {args.model}")
    logs, per_k = characterize(
        items, generate, extract, templates, args.branching, args.n_steps, args.rollouts
    )
    stats = statistics(logs, per_k)

    print(f"\nnodes logged: {len(logs) * args.branching}  problems: {len(items)}")
    print(f"  cheap-vs-true Spearman rho = {stats['spearman']:.3f}  Pearson r = {stats['pearson']:.3f}")
    print(f"  edge-following hit rate, all steps (chance {1/args.branching:.2f}) = {stats['hit_all']:.3f}")
    lo, hi = stats["hit_pivotal_ci"]
    print(
        f"  edge-following hit rate, pivotal steps = {stats['hit_pivotal']:.3f} "
        f"[{lo:.2f}, {hi:.2f}] (n={stats['n_pivotal']})"
    )
    print(f"  mean true value lost per step (edge gap) = {stats['mean_gap']:.3f}")
    print(
        "  pivotal-step fraction: "
        + "; ".join(f"tau={t}: {v:.0%}" for t, v in stats["pivotal_fractions"].items())
    )
    print(f"  mean pivotal steps per problem (K at tau={PIVOTAL_TAU}) = {stats['mean_k']:.2f}")
    _plot_and_table(stats, per_k, args.benchmark)
    print(f"  client stats: {client.stats()}")


def _plot_and_table(stats, per_k, benchmark: str) -> None:
    import matplotlib.pyplot as plt  # noqa: E402

    from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style  # noqa: E402

    set_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    axA.scatter(stats["cheap"], stats["true"], s=12, alpha=0.35, color=PALETTE["blue"])
    coef = np.polyfit(stats["cheap"], stats["true"], 1)
    xs = np.linspace(0, 1, 20)
    axA.plot(xs, np.polyval(coef, xs), color=PALETTE["orange"], lw=2, label="linear fit")
    axA.set_xlabel("cheap probe value (self-consistency)")
    axA.set_ylabel("true node value (fraction of rollouts correct)")
    axA.set_title(
        f"Cheap probe tracks true value "
        f"(Spearman $\\rho$={stats['spearman']:.2f}; edge-hit {stats['hit_all']:.2f})"
    )
    axA.legend(loc="upper left", fontsize=8)

    axB.hist(per_k, bins=np.arange(-0.5, per_k.max() + 1.5), color=PALETTE["green"])
    axB.set_xlabel(f"pivotal steps per problem (spread > {PIVOTAL_TAU})")
    axB.set_ylabel("number of problems")
    axB.set_title("Most steps are smooth; few are pivotal (violations)")

    fig.suptitle(f"Reasoning value function as almost tree-$K$-Lipschitz ({benchmark})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = save_figure(fig, f"reasoning_tree_lipschitz_{benchmark}")
    print(f"saved chart to {out} (+ .png)")

    lo, hi = stats["hit_pivotal_ci"]
    rows = [
        ("Cheap-vs-true node value (Spearman $\\rho$)", f"{stats['spearman']:.3f}"),
        ("Cheap-vs-true node value (Pearson $r$)", f"{stats['pearson']:.3f}"),
        ("Edge-following hit rate, all steps", f"{stats['hit_all']:.3f}"),
        (
            "Edge-following hit rate, pivotal steps only",
            f"{stats['hit_pivotal']:.3f} [{lo:.2f}, {hi:.2f}] ($n={stats['n_pivotal']}$)",
        ),
        ("Mean true value lost per step (edge gap)", f"{stats['mean_gap']:.3f}"),
        (
            "Pivotal-step fraction",
            "; ".join(f"$\\tau{{=}}{t}$: {v:.0%}".replace("%", "\\%") for t, v in stats["pivotal_fractions"].items()),
        ),
    ]
    tex = (
        f"% {benchmark} tree-Lipschitz characterization (auto-generated)\n"
        "\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
        + "\n".join(f"{a} & {b} \\\\" for a, b in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / f"reasoning_tree_lipschitz_{benchmark}_table.tex").write_text(tex)
    print(f"wrote LaTeX table to {FIGURE_DIR}/reasoning_tree_lipschitz_{benchmark}_table.tex")


if __name__ == "__main__":
    main()
