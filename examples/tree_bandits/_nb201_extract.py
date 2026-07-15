"""One-off: extract NAS-Bench-201 test accuracies into an encoding-ordered leaf-means cache.

NAS-Bench-201's search space is a cell of 4 nodes / 6 edges, each edge taking one of 5
operations, giving 5^6 = 15,625 architectures. The 6-operation code is therefore a natural
branching-5, depth-6 tree: fixing the first l edge-operations names a level-l subtree, whose
value is the mean accuracy over its 5^(6-l) completions. We store, per dataset, the 15,625
final test accuracies ordered by this base-5 encoding, so a TreeBandit built on the array has
the architecture search space itself as its tree. Cheap internal probe = the average accuracy
of a partially-specified architecture family (biased proxy for its best completion); expensive
leaf = the full-training accuracy of one architecture.

Run on the box:  ~/canopy/.venv/bin/python examples/tree_bandits/_nb201_extract.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "nasbench201"
PTH = CACHE / "nb201_v1_0.pth"
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OP_INDEX = {op: i for i, op in enumerate(OPS)}
DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]


def arch_to_ops(arch_str: str) -> list[int]:
    """Parse '|op~0|+|op~0|op~1|+|op~0|op~1|op~2|' into its 6 operation indices (edge order)."""
    ops = []
    for block in arch_str.split("+"):
        for tok in block.strip("|").split("|"):
            name = tok.split("~")[0]
            ops.append(OP_INDEX[name])
    return ops  # length 6


def leaf_index(ops: list[int]) -> int:
    """Base-5 encoding of the 6 edge-operations -> leaf index in a branching-5, depth-6 tree."""
    idx = 0
    for o in ops:
        idx = idx * 5 + o
    return idx


def main() -> None:
    import torch  # NAS-Bench-201 is an official checkpoint; allow full unpickle on torch>=2.6

    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    from nas_201_api import NASBench201API as API

    api = API(str(PTH), verbose=False)
    n = len(api)
    print(f"loaded NAS-Bench-201: {n} architectures")
    assert n == 5**6, f"expected 15625 archs, got {n}"

    acc = {d: np.full(5**6, np.nan) for d in DATASETS}
    for i, arch_str in enumerate(api):
        li = leaf_index(arch_to_ops(arch_str))
        for d in DATASETS:
            info = api.get_more_info(i, d, hp="200", is_random=False)
            # test accuracy keys differ slightly by dataset; try the common ones
            a = info.get("test-accuracy")
            if a is None:
                a = info.get("ori-test-accuracy") or info.get("x-test-accuracy")
            acc[d][li] = float(a)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}")

    for d in DATASETS:
        assert not np.isnan(acc[d]).any(), f"missing accuracies for {d}"
    out = CACHE / "nb201_leaf_means.npz"
    np.savez_compressed(out, **{d: acc[d] / 100.0 for d in DATASETS})  # store as [0,1]
    print(f"saved {out}")
    for d in DATASETS:
        v = acc[d] / 100.0
        print(f"  {d}: mean={v.mean():.3f} max={v.max():.3f} min={v.min():.3f}")


if __name__ == "__main__":
    main()
