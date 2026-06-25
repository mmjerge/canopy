# Long-horizon agentic search: rollout-guided planning vs. best-of-N

This is the agentic / multi-step extension of the reasoning-tree result. It moves from a
static tree of leaf rewards (`oco.bandits.reasoning`) to a **real sequential decision process
with state and dynamics** (`oco.bandits.agentic`), where the horizon is genuinely long and the
cheap value signal lives at *intermediate* states.

## Why move off GSM8K

GSM8K turned out to be the wrong arena: at a matched ~29-call budget on Llama-3.1-8B over 50
problems, value-guided search (0.82) **did not** beat best-of-N self-consistency (0.88), and
an oracle-value arm (0.86) also failed to beat it. The diagnosis is structural, not a bug:

- GSM8K has **no real horizon** -- the only signal is the final number, which is exactly what
  best-of-N already majority-votes over. There is no intermediate state to prune on.
- The benchmark is **saturated**: best-of-N solves ~88%, and the misses are problems the model
  cannot solve at all, so reallocating search cannot help.

The lesson: value-guided search beats best-of-N only where (1) the horizon is long enough that
landing on a full correct trajectory by sampling is unlikely, and (2) a cheap probe is
informative at intermediate states. That is a property of the *task*, and GSM8K lacks it.

## The environment

`GridWorld` is deterministic grid navigation: reach the goal corner from the start corner
within `horizon` steps, four moves, walls/edges block. Terminal reward is
`1 - manhattan(state, goal) / d_max` in `[0, 1]`, equal to `1` only at the goal. This smooth
shaping is what makes a **random rollout** from a state a weak-but-real estimate of that
state's distance to the goal -- the cheap, biased multi-fidelity probe, the direct analogue of
probing an internal node in `TreeBandit`.

## The two planners (matched step budget)

- **`best_of_n_plan`** -- structure-blind random shooting: sample `n` whole random
  trajectories, keep the best by terminal reward. To land on a length-`H` correct path it must
  *sample* one, which is exponentially unlikely as `H` grows.
- **`rollout_policy_plan`** -- edge-following: at each state, score each action by a few random
  rollouts to the horizon (the probe) and commit to the best, then advance. One decision at a
  time, so cost grows polynomially. (One-step rollout policy improvement / depth-one tree
  search with random-rollout value.)

Budget is measured in **simulator transitions**. `compare_matched_budget` runs value-guided
first, then gives best-of-N `round(vg_steps / horizon)` trajectories so both spend ~the same
number of steps -- the same matched-budget discipline used for the GSM8K calls.

## Result (synthetic environment, fully reproducible)

`success_rates` over 60 seeds, `n_rollouts=8`, at a matched step budget:

| grid | horizon | rollout-guided | best-of-N |
|------|---------|----------------|-----------|
| 4    | 10      | 0.98           | 1.00      |
| 6    | 16      | 0.93           | 0.70      |
| 8    | 22      | 0.87           | 0.22      |
| 10   | 28      | 0.73           | 0.05      |
| 12   | 34      | 0.73           | 0.05      |
| 14   | 40      | 0.67           | 0.00      |

At short horizons best-of-N ties or wins (random shooting is fine when the path is short). As
the horizon grows it collapses to zero while rollout-guided planning holds up -- the
long-horizon separation, far outside noise, and the opposite of the GSM8K picture. The chart is
`examples/tree_agentic_search.png`.

## Honest scope and limits

- This is a **search / compute-allocation** result on a **synthetic environment**, not an LLM
  benchmark. It shows the mechanism works when its assumptions hold; it does not by itself
  claim an LLM win.
- It needs the value probe to be **informative**. In an open grid the distance shaping makes
  random rollouts informative. Add a maze with a local optimum (a dead-end that looks closer in
  Manhattan distance) and the probe becomes misleading -- the gap closes. This mirrors the
  `adversarial_spike` vs. `hierarchical_gaussian` contrast in `TreeBandit`: structure helps
  exactly when the value is structured.
- The natural next step is to put a real LLM policy/value in this loop on a long-horizon
  agentic benchmark (ALFWorld / WebArena-style), where the intermediate signal is real and
  best-of-N over whole trajectories is known to be wasteful.
