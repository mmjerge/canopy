# Pre-registration: retrieval-scored trimming on LongBench

Committed before any run. Results reported in the paper regardless of outcome.

## Motivation and hypothesis

The LongBench per-item oracle attains 0.55 QA-F1 at 39% of the tokens, above full context
(0.44), while prefix trimming loses F1 monotonically as tokens are cut. The gap suggests
the answer evidence is dispersed through the document, so which tokens are kept matters
more than how many. Hypothesis (directional): replacing the prefix-truncation primitive
with retrieval-scored chunk selection (keep the chunks most lexically relevant to the
question, at the same token budget) yields higher QA-F1 at matched tokens than prefix
trimming, closing part of the oracle gap. The bandit machinery (arms = keep fractions,
regions = tasks, same lambda sweep) is unchanged; only the trimming primitive changes.

## Protocol (fixed in advance)

- Same 180 items (6 tasks x 30), same model (us.amazon.nova-lite-v1:0), same
  keep-fraction arms, same lambda grid, same F1 grader as the existing prefix-mode run.
- Retrieval primitive: split context into 64-word chunks, score each chunk by normalized
  question-term overlap, keep highest-scoring chunks until the keep-fraction word budget
  is reached, reassemble kept chunks in original document order. Deterministic, no new
  dependencies, fixed before running.
- Outputs written alongside (not over) the prefix-mode artifacts.
- Primary endpoint: F1 at matched tokens vs the prefix-mode adaptive frontier, and the
  fraction of the (oracle - adaptive) gap closed at the lambda = 0.3 operating point.
- Both primitives will be reported side by side; if retrieval trimming is worse or flat,
  that is the result.
