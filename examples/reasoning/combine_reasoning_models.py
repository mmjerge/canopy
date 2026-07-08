r"""Combine per-model reasoning-search runs into one comparison figure.

Reads every ``reasoning_search_<bench>[_<tag>]_results.json`` in ``paper/figures/`` (the headline
run plus any ``--tag`` model runs from ``reasoning_search.py``) and plots value-guided vs. best-of-N
across models. The scientific point: the theory predicts the value-guided gain is *largest
mid-capability* (reachable-but-unreliable = many decision steps K) and *shrinks* toward zero on
frontier models (near-saturated, small K) and on weak models (a correct trace is unreachable, so
no reallocation of compute helps). This turns the multi-model runs into a single panel testing that
predicted shape, with bootstrap CIs.

For each model we take the largest matched-compute budget level and bootstrap over problems
(paired, since both strategies see the same problems) for the accuracy CIs and the gap Delta.

Run (after the per-model runs finish):
    uv run --extra plot python examples/reasoning/combine_reasoning_models.py --benchmark math
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style  # noqa: E402


def _label(path: Path, bench: str, model: str) -> str:
    """Human label for a run: the filename tag if present, else a short form of the model id."""
    stem = path.name[: -len("_results.json")]
    prefix = f"reasoning_search_{bench}"
    tag = stem[len(prefix):].lstrip("_")
    if tag:
        return tag
    return model.split(".")[-1].replace("-instruct", "").replace("-v1:0", "")[:18] or "headline"


def _max_budget_level(levels: dict) -> dict:
    """The level with the largest matched compute budget (most informative comparison point)."""
    return max(levels.values(), key=lambda lv: lv["matched_budget"])


def _paired_bootstrap(bo_hits, vg_hits, iters=4000, seed=0):
    """Bootstrap over problems; return (bo_mean,bo_lo,bo_hi, vg_mean,vg_lo,vg_hi, d_mean,d_lo,d_hi)."""
    bo, vg = np.asarray(bo_hits, float), np.asarray(vg_hits, float)
    n = bo.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    bo_s, vg_s = bo[idx].mean(1), vg[idx].mean(1)
    d_s = vg_s - bo_s
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return (float(bo.mean()), *q(bo_s), float(vg.mean()), *q(vg_s),
            float((vg - bo).mean()), *q(d_s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="math")
    ap.add_argument("--exclude", default="",
                    help="comma-separated model labels to drop (e.g. models that did not follow "
                         "the answer format on this benchmark, so their scores are "
                         "extraction-confounded rather than a real capability measurement)")
    args = ap.parse_args()
    bench = args.benchmark
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    files = sorted(FIGURE_DIR.glob(f"reasoning_search_{bench}*_results.json"))
    rows = []
    for f in files:
        data = json.loads(f.read_text())
        levels = data.get("levels", {})
        if not levels:
            continue
        label = _label(f, bench, data.get("model", ""))
        if label in exclude:
            print(f"  (excluding {label}: format non-compliance on {bench.upper()})")
            continue
        lv = _max_budget_level(levels)
        bo_hits, vg_hits = lv["best_of_n"]["hits"], lv["value_guided"]["hits"]
        if not bo_hits:
            continue
        stats = _paired_bootstrap(bo_hits, vg_hits)
        rows.append({
            "label": label,
            "n": len(bo_hits), "budget": lv["matched_budget"],
            "bo": stats[0], "bo_lo": stats[1], "bo_hi": stats[2],
            "vg": stats[3], "vg_lo": stats[4], "vg_hi": stats[5],
            "d": stats[6], "d_lo": stats[7], "d_hi": stats[8],
        })
    if not rows:
        print(f"No reasoning_search_{bench}*_results.json files found in {FIGURE_DIR}. "
              "Run reasoning_search.py (headline + --tag runs) first.")
        return

    rows.sort(key=lambda r: r["bo"])  # order by capability (best-of-N accuracy proxy)
    print(f"{bench.upper()} across {len(rows)} models (largest-budget level, 95% CI over problems):")
    for r in rows:
        print(f"  {r['label']:20s} best-of-N={r['bo']:.3f} [{r['bo_lo']:.2f},{r['bo_hi']:.2f}]  "
              f"value-guided={r['vg']:.3f} [{r['vg_lo']:.2f},{r['vg_hi']:.2f}]  "
              f"Delta={r['d']:+.3f} [{r['d_lo']:+.2f},{r['d_hi']:+.2f}]")

    _write_table(rows, bench)
    _plot(rows, bench)


def _write_table(rows, bench):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    body = []
    for r in rows:
        body.append(
            f"\\texttt{{{r['label'].replace('_', chr(92) + '_')}}} & "
            f"{r['bo']:.3f} [{r['bo_lo']:.2f}, {r['bo_hi']:.2f}] & "
            f"{r['vg']:.3f} [{r['vg_lo']:.2f}, {r['vg_hi']:.2f}] & "
            f"${r['d']:+.3f}$ [{r['d_lo']:+.2f}, {r['d_hi']:+.2f}] \\\\"
        )
    tex = (
        f"% {bench.upper()} value-guided vs best-of-N across models "
        "(auto-generated by combine_reasoning_models.py)\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Model & best-of-N [95\\% CI] & value-guided [95\\% CI] & $\\Delta$ [95\\% CI] \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / f"reasoning_models_{bench}_table.tex").write_text(tex)


def _plot(rows, bench):
    set_style()
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    bo = np.array([r["bo"] for r in rows])
    vg = np.array([r["vg"] for r in rows])
    bo_err = np.array([[r["bo"] - r["bo_lo"] for r in rows], [r["bo_hi"] - r["bo"] for r in rows]])
    vg_err = np.array([[r["vg"] - r["vg_lo"] for r in rows], [r["vg_hi"] - r["vg"] for r in rows]])
    d = np.array([r["d"] for r in rows])
    d_err = np.array([[r["d"] - r["d_lo"] for r in rows], [r["d_hi"] - r["d"] for r in rows]])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    axA.errorbar(x - 0.08, bo, yerr=bo_err, fmt="s", color=PALETTE["orange"], capsize=3,
                 label="best-of-N")
    axA.errorbar(x + 0.08, vg, yerr=vg_err, fmt="o", color=PALETTE["red"], capsize=3,
                 label="value-guided (ours)")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=30, ha="right")
    axA.set_ylabel(f"{bench.upper()} accuracy (largest budget)")
    axA.set_title("Value-guided vs. best-of-N across models")
    axA.legend(loc="best")

    axB.axhline(0.0, color=PALETTE["gray"], lw=1, ls="--")
    axB.errorbar(bo, d, yerr=d_err, fmt="o", color=PALETTE["blue"], capsize=3)
    for r in rows:
        axB.annotate(r["label"], (r["bo"], r["d"]), fontsize=7, xytext=(4, 3),
                     textcoords="offset points")
    axB.set_xlabel("model capability (best-of-N accuracy)")
    axB.set_ylabel("value-guided gain $\\Delta$")
    axB.set_title("Gain vs. capability (theory: peaks mid-capability)")

    fig.suptitle(f"Value-guided search vs. best-of-N across models ({bench.upper()})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = save_figure(fig, f"reasoning_models_{bench}")
    print(f"\nwrote table + figure ({out}, +.png) to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
