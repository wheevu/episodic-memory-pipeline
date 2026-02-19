"""Pytest configuration and shared fixtures."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.storage import LanceStore


def pytest_addoption(parser: Any) -> None:
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model downloads",
    )


def pytest_configure(config: Any) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (require model downloads, may take > 5s)"
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip slow tests unless explicitly enabled."""
    if config.getoption("--run-slow"):
        return

    markexpr = config.getoption("-m", default="")
    if markexpr and "slow" in markexpr and "not slow" not in markexpr:
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def temp_dir() -> Path:
    """Create and clean up a temporary directory."""
    tmp = Path(tempfile.mkdtemp(prefix="epmem_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def lance_store(temp_dir: Path) -> LanceStore:
    """Create a temporary LanceStore fixture."""
    return LanceStore(temp_dir / "lancedb", embedding_dimension=384)
