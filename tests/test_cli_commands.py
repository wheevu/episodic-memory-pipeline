"""CLI behavior tests for command/service contracts."""

from pathlib import Path

from click.testing import CliRunner

from config import Config
from src.bootstrap import get_components
from src.cli import app, reset_components


def _test_components(tmp_path: Path):
    cfg = Config()
    cfg.lance_db_path = tmp_path / "lancedb"
    cfg.embedding_provider = "mock"
    cfg.embedding_dimension = 384
    cfg.llm_provider = "ollama"
    return get_components(config=cfg, force_mock=True, verbose=False)


def test_stats_command_renders_vector_table(monkeypatch, tmp_path: Path) -> None:
    """`stats` should render vector stats without crashing."""
    components = _test_components(tmp_path)

    def fake_get_pipeline_components(use_mock: bool = False):
        return components

    monkeypatch.setattr("src.cli.get_pipeline_components", fake_get_pipeline_components)

    runner = CliRunner()
    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0, result.output
    assert "Database Statistics" in result.output
    assert "Vector Store Statistics" in result.output
    assert "episodes" in result.output
    assert "facts" in result.output
    assert "summaries" in result.output

    reset_components()
