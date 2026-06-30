"""Smooth-tree bandits: multi-fidelity tree environment, identification, and regret.

The adversarial multi-armed bandit (EXP3) plus the tree-structured pure-exploration and
regret-minimization environment with cheap/biased internal probes and expensive/unbiased
leaf evaluations, and the applied instantiations (LLM routing, prefix caching, test-time
search).
"""

from canopy.bandits.exp3 import EXP3
from canopy.bandits.baselines import SuccessiveEliminationTopK
from canopy.bandits.online import (
    RegretResult,
    ViolationReport,
    MultiscaleEdgeMap,
    detect_violations,
    multiscale_edge_map,
    run_adaptive,
    run_adaptive_mgf,
    run_adaptive_variance,
    run_fixed_depth,
    run_hoo,
    run_hybrid,
    run_local_lipschitz,
)
from canopy.bandits.routing import PrefixTreeRouting, RouteResult, make_routing_scenario, run_router
from canopy.bandits.prefix_cache import CacheResult, PrefixCacheEnv, offline_optimal, run_cache
from canopy.bandits.reasoning import (
    best_of_n,
    reasoning_tree_rewards,
    success_rate,
    value_guided_search,
)
from canopy.bandits.reasoning_llm import (
    Budget,
    SearchResult,
    extract_answer,
    is_correct,
)
from canopy.bandits.reasoning_llm import best_of_n as best_of_n_llm
from canopy.bandits.reasoning_llm import value_guided_search as value_guided_search_llm
from canopy.bandits.agentic import (
    GridWorld,
    PlanResult,
    best_of_n_plan,
    compare_matched_budget,
    rollout_policy_plan,
    success_rates,
)
from canopy.bandits.agentic_llm import (
    AgentEnv,
    EpisodeResult,
    ReplayCloneEnv,
    StepEnv,
    best_of_n_episodes,
    value_guided_episode,
)
from canopy.bandits.agentic_llm import compare_matched_budget as compare_matched_budget_agent
from canopy.bandits.rewards import (
    adversarial_spike_leaf_means,
    geometric_sigma,
    heterogeneous_smoothness_leaf_means,
    hierarchical_gaussian_leaf_means,
    hierarchical_spread,
    lipschitz_spread,
    piecewise_smooth_leaf_means,
    violation_family_leaf_means,
)
from canopy.bandits.topk import HierarchicalTopK, TopKResult, UniformTopK
from canopy.bandits.tree import Node, TreeBandit

__all__ = [
    "EXP3",
    "TreeBandit",
    "Node",
    "UniformTopK",
    "HierarchicalTopK",
    "SuccessiveEliminationTopK",
    "TopKResult",
    "RegretResult",
    "ViolationReport",
    "detect_violations",
    "MultiscaleEdgeMap",
    "multiscale_edge_map",
    "run_adaptive",
    "run_adaptive_mgf",
    "run_adaptive_variance",
    "run_fixed_depth",
    "run_hoo",
    "run_hybrid",
    "run_local_lipschitz",
    "geometric_sigma",
    "hierarchical_gaussian_leaf_means",
    "heterogeneous_smoothness_leaf_means",
    "hierarchical_spread",
    "lipschitz_spread",
    "adversarial_spike_leaf_means",
    "piecewise_smooth_leaf_means",
    "violation_family_leaf_means",
    "PrefixTreeRouting",
    "RouteResult",
    "make_routing_scenario",
    "run_router",
    "PrefixCacheEnv",
    "CacheResult",
    "run_cache",
    "offline_optimal",
    "reasoning_tree_rewards",
    "best_of_n",
    "value_guided_search",
    "success_rate",
    "extract_answer",
    "is_correct",
    "Budget",
    "SearchResult",
    "best_of_n_llm",
    "value_guided_search_llm",
    "GridWorld",
    "PlanResult",
    "best_of_n_plan",
    "rollout_policy_plan",
    "compare_matched_budget",
    "success_rates",
    "AgentEnv",
    "EpisodeResult",
    "ReplayCloneEnv",
    "StepEnv",
    "best_of_n_episodes",
    "value_guided_episode",
    "compare_matched_budget_agent",
]
