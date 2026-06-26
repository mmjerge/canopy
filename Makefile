.PHONY: help install lint format typecheck test cov build figures clean

help:
	@echo "Targets:"
	@echo "  install    install the package with dev + plot extras (editable)"
	@echo "  lint       run pre-commit (black, flake8, mypy, hygiene) + ruff"
	@echo "  format     auto-format with black"
	@echo "  typecheck  run mypy"
	@echo "  test       run the test suite"
	@echo "  cov        run tests with branch coverage (fails under 90%)"
	@echo "  build      build the sdist and wheel"
	@echo "  figures    regenerate the synthetic-demo charts into examples/images/"
	@echo "  clean      remove build/coverage artifacts"

install:
	pip install -e ".[dev,plot]"

lint:
	pre-commit run --all-files
	ruff check src tests examples

format:
	black src tests examples

typecheck:
	mypy

test:
	pytest

cov:
	pytest --cov=canopy --cov-branch --cov-report=term-missing --cov-fail-under=90

build:
	python -m build

# Regenerate the synthetic (no-credentials) demo charts.
figures:
	python examples/tree_bandits/benchmark.py
	python examples/tree_bandits/regret_storage_demo.py
	python examples/tree_bandits/violation_regret_demo.py
	python examples/tree_bandits/infinite_depth_demo.py
	python examples/tree_bandits/local_lipschitz_demo.py
	python examples/tree_bandits/jump_robustness_demo.py
	python examples/tree_bandits/multiscale_edge_demo.py
	python examples/tree_bandits/targeted_sampling_demo.py
	python examples/reasoning/reasoning_search_demo.py
	python examples/reasoning/agentic_search_demo.py
	python examples/llm_routing/llm_routing_demo.py
	python examples/llm_routing/prefix_cache_demo.py

clean:
	rm -rf dist build .coverage coverage.xml htmlcov .mypy_cache .pytest_cache .ruff_cache
