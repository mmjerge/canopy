"""Smoke tests: every example module must import cleanly.

The demos aren't otherwise exercised by the test suite, so import moves or API
changes can silently break them (as nearly happened during the examples
restructure). Importing each module runs its top-level code (imports, constants,
function definitions) without invoking ``main()`` (guarded by ``__main__``), which
catches import/syntax/path errors cheaply.

Requires matplotlib (several demos import it at module level); skipped otherwise.
The headless Agg backend avoids needing a display.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "path", EXAMPLE_FILES, ids=[str(p.relative_to(EXAMPLES_DIR)) for p in EXAMPLE_FILES]
)
def test_example_module_imports(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec (canonical importlib pattern): dataclasses and other tools resolve
    # annotations via sys.modules[cls.__module__], which fails if the module isn't registered.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
