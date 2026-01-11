"""
Pytest configuration and fixtures for episodic-memory-pipeline tests.

Test Categories:
================

1. Fast tests (default):
   - Use mock providers only
   - No network access required
   - No model downloads required
   - Run with: pytest

2. Slow tests (@pytest.mark.slow):
   - May download SentenceTransformer models
   - May take > 5 seconds
   - Run with: pytest -m slow
   - Or run all: pytest --run-slow

3. Tests requiring FAISS (@pytest.mark.requires_faiss):
   - Automatically skipped if faiss-cpu is not installed
   - Most pipeline tests fall into this category

IMPORTANT: macOS FAISS/SentenceTransformers Note
================================================
On macOS, there's a known issue where FAISS and SentenceTransformers can
conflict during Python cleanup, causing segfaults. The bootstrap module
(src/bootstrap.py) handles this for CLI usage.

For tests, we recommend:
- Running slow embedding tests separately: pytest -m slow tests/test_embeddings.py
- Running fast tests normally: pytest

See ARCHITECTURE.md for more details on the initialization order constraint.
"""
import pytest
from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Add custom command-line options.

    Args:
        parser: Pytest option parser.

    Returns:
        None.
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model downloads"
    )


def pytest_configure(config: Any) -> None:
    """Configure pytest markers.

    Args:
        config: Pytest configuration object.

    Returns:
        None.
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (require model downloads, may take > 5s)"
    )
    config.addinivalue_line(
        "markers",
        "requires_faiss: marks tests that require FAISS to be installed"
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """
    Modify test collection based on markers and options.
    
    By default (without --run-slow), slow tests are skipped.
    With --run-slow, all tests run.
    With -m slow, only slow tests run.
    
    Tests marked with requires_faiss are automatically skipped if faiss is not available.

    Args:
        config: Pytest configuration object.
        items: Collected test items to potentially modify.

    Returns:
        None.
    """
    # Check for FAISS availability
    try:
        import faiss
        has_faiss = True
    except ImportError:
        has_faiss = False
    
    # Handle slow tests
    if config.getoption("--run-slow"):
        # --run-slow given: don't skip slow tests
        pass
    else:
        # Check if user explicitly requested slow tests with -m slow
        markexpr = config.getoption("-m", default="")
        if not (markexpr and "slow" in markexpr and "not slow" not in markexpr):
            # Skip slow tests by default
            skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
            for item in items:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)
    
    # Handle FAISS-dependent tests
    if not has_faiss:
        skip_faiss = pytest.mark.skip(reason="FAISS not installed (pip install faiss-cpu)")
        for item in items:
            if "requires_faiss" in item.keywords:
                item.add_marker(skip_faiss)

