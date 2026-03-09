"""Bootstrap tests for embedding/store dimension consistency."""

from pathlib import Path

import pytest

from config import Config
from src.bootstrap import get_components
from src.embeddings import MockEmbeddingProvider


def _base_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.lance_db_path = tmp_path / "lancedb"
    cfg.embedding_provider = "local"
    cfg.llm_provider = "openai"
    cfg.openai_api_key = None
    return cfg


def test_bootstrap_fails_when_explicit_dimension_mismatches_provider(
    monkeypatch, tmp_path: Path
) -> None:
    """Explicit EMBEDDING_DIMENSION should fail fast on mismatch."""

    def fake_get_embedding_provider(*args, **kwargs):
        return MockEmbeddingProvider(dimension=256)

    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setattr("src.bootstrap.get_embedding_provider", fake_get_embedding_provider)

    cfg = _base_config(tmp_path)
    cfg.embedding_dimension = 1024

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION"):
        get_components(config=cfg, force_mock=False, verbose=False)


def test_bootstrap_auto_adjusts_dimension_when_not_explicit(monkeypatch, tmp_path: Path) -> None:
    """When dimension is not explicitly pinned, bootstrap should align to provider."""

    def fake_get_embedding_provider(*args, **kwargs):
        return MockEmbeddingProvider(dimension=256)

    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    monkeypatch.setattr("src.bootstrap.get_embedding_provider", fake_get_embedding_provider)

    cfg = _base_config(tmp_path)
    cfg.embedding_dimension = 1024

    components = get_components(config=cfg, force_mock=False, verbose=False)

    assert components.config.embedding_dimension == 256
    assert components.lance_store.embedding_dimension == 256
