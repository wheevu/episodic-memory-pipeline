"""
Acceptance tests for the episodic memory pipeline.

These tests verify end-to-end user expectations:
- Fresh clone can bootstrap and run demo
- Doctor command works without network
- Eval runs are reproducible
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestBootstrapDemo:
    """Test that fresh clone bootstrap works correctly."""
    
    @pytest.fixture
    def temp_data_dir(self) -> Path:
        """Create a temporary data directory.

        Returns:
            A temporary directory path used as a data directory.
        """
        tmp = tempfile.mkdtemp(prefix="episodic_test_")
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    
    @pytest.fixture
    def fixtures_path(self) -> Path:
        """Return the path to the demo fixtures JSON file.

        Returns:
            Path to `demo_data/fixtures.json`.
        """
        return PROJECT_ROOT / "demo_data" / "fixtures.json"
    
    def test_fixtures_file_exists(self, fixtures_path: Path) -> None:
        """Verify the demo fixtures file exists.

        Args:
            fixtures_path: Path to the fixtures file.

        Returns:
            None.
        """
        assert fixtures_path.exists(), f"Fixtures file not found: {fixtures_path}"
    
    def test_fixtures_file_valid_json(self, fixtures_path: Path) -> None:
        """Verify fixtures file is valid JSON and has required top-level keys.

        Args:
            fixtures_path: Path to the fixtures file.

        Returns:
            None.
        """
        with open(fixtures_path) as f:
            data = json.load(f)
        
        assert "version" in data
        assert "episodes" in data
        assert len(data["episodes"]) > 0
    
    def test_fixtures_episodes_have_required_fields(self, fixtures_path: Path) -> None:
        """Verify each fixture episode has required fields and reasonable length.

        Args:
            fixtures_path: Path to the fixtures file.

        Returns:
            None.
        """
        with open(fixtures_path) as f:
            data = json.load(f)
        
        for i, ep in enumerate(data["episodes"]):
            assert "text" in ep, f"Episode {i} missing 'text' field"
            assert "source" in ep, f"Episode {i} missing 'source' field"
            assert len(ep["text"]) > 10, f"Episode {i} text too short"
    
    def test_bootstrap_creates_artifacts(self, temp_data_dir: Path, fixtures_path: Path) -> None:
        """Test that bootstrap creates expected artifacts on disk.

        Args:
            temp_data_dir: Temporary data directory.
            fixtures_path: Path to fixtures file.

        Returns:
            None.
        """
        from scripts.bootstrap_demo import bootstrap_demo
        
        summary = bootstrap_demo(
            fixtures_path=fixtures_path,
            data_dir=temp_data_dir,
            use_mock=True,  # Use mock for fast testing
            dry_run=False,
            quiet=True
        )
        
        # Check summary
        assert summary["episodes_ingested"] > 0, "No episodes ingested"
        assert len(summary["errors"]) == 0, f"Errors: {summary['errors']}"
        
        # Check artifacts created
        assert (temp_data_dir / "memory.db").exists(), "Database not created"
        
        # Check for FAISS files
        faiss_files = list(temp_data_dir.glob("*.faiss"))
        assert len(faiss_files) > 0, "No FAISS index files created"
        
        # Check for ID map files
        npy_files = list(temp_data_dir.glob("*.npy"))
        assert len(npy_files) > 0, "No ID map files created"
    
    def test_bootstrap_dry_run_no_changes(self, temp_data_dir: Path, fixtures_path: Path) -> None:
        """Test that dry-run does not create any files.

        Args:
            temp_data_dir: Temporary data directory.
            fixtures_path: Path to fixtures file.

        Returns:
            None.
        """
        from scripts.bootstrap_demo import bootstrap_demo
        
        summary = bootstrap_demo(
            fixtures_path=fixtures_path,
            data_dir=temp_data_dir,
            use_mock=True,
            dry_run=True,
            quiet=True
        )
        
        # Should report what would be done
        assert summary["episodes_ingested"] > 0
        
        # But no files should be created
        assert not (temp_data_dir / "memory.db").exists()
        assert len(list(temp_data_dir.glob("*.faiss"))) == 0
    
    def test_bootstrap_clean_removes_existing(self, temp_data_dir: Path, fixtures_path: Path) -> None:
        """Test that cleaning removes existing bootstrap artifacts.

        Args:
            temp_data_dir: Temporary data directory.
            fixtures_path: Path to fixtures file.

        Returns:
            None.
        """
        from scripts.bootstrap_demo import bootstrap_demo, clean_data_directory
        
        # First bootstrap
        bootstrap_demo(
            fixtures_path=fixtures_path,
            data_dir=temp_data_dir,
            use_mock=True,
            quiet=True
        )
        
        # Verify files exist
        assert (temp_data_dir / "memory.db").exists()
        
        # Clean
        clean_data_directory(temp_data_dir, dry_run=False, quiet=True)
        
        # Files should be removed
        assert not (temp_data_dir / "memory.db").exists()


class TestDoctorCommand:
    """Test that doctor command works correctly."""
    
    def test_dry_diagnostics_no_initialization(self) -> None:
        """Test dry-run diagnostics doesn't initialize components.

        Returns:
            None.
        """
        from src.services.diagnostics import DiagnosticsService
        
        service = DiagnosticsService()
        
        # Should work without components
        result = service.run_dry_diagnostics(force_mock=False)
        
        assert "env_vars" in result
        assert "resolved" in result
        assert "predictions" in result
        assert "suggestions" in result
    
    def test_dry_diagnostics_detects_mock_status(self) -> None:
        """Test that dry diagnostics correctly predicts mock usage.

        Returns:
            None.
        """
        from src.services.diagnostics import DiagnosticsService
        
        service = DiagnosticsService()
        
        # With force_mock=True
        result = service.run_dry_diagnostics(force_mock=True)
        assert result["predictions"]["will_use_mock_embeddings"] == True
        assert result["predictions"]["will_use_mock_llm"] == True
    
    def test_dry_diagnostics_provides_suggestions_for_mock(self) -> None:
        """Test that suggestions are provided when mock would be used.

        Returns:
            None.
        """
        from src.services.diagnostics import DiagnosticsService
        
        service = DiagnosticsService()
        
        # Ensure no API key is set
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            result = service.run_dry_diagnostics(force_mock=False)
        
        # Should have suggestions if mock would be used
        if result["predictions"]["will_use_mock_llm"]:
            assert len(result["suggestions"]) > 0


class TestEvaluationReproducibility:
    """Test that evaluation runs are reproducible."""
    
    @pytest.fixture
    def temp_runs_dir(self) -> Path:
        """Create a temporary runs directory.

        Returns:
            A temporary directory path used as the evaluation runs directory.
        """
        tmp = tempfile.mkdtemp(prefix="episodic_eval_")
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    
    def test_eval_run_saves_to_disk(self, temp_runs_dir: Path) -> None:
        """Test that eval run saves results to disk.

        Args:
            temp_runs_dir: Temporary runs directory.

        Returns:
            None.
        """
        from src.services.evaluation import EvaluationService
        from src.bootstrap import get_components
        from config import config
        
        components = get_components(config=config, force_mock=True, verbose=False)
        service = EvaluationService(components, runs_dir=temp_runs_dir)
        
        result = service.run_evaluation(
            scenario="diary",
            precision_k=5,
            save=True
        )
        
        assert result.success, f"Eval failed: {result.error}"
        
        # Check file was saved
        run_dir = temp_runs_dir / result.run_id
        assert run_dir.exists(), "Run directory not created"
        assert (run_dir / "eval_run.json").exists(), "eval_run.json not created"
    
    def test_eval_run_schema_has_required_fields(self, temp_runs_dir: Path) -> None:
        """Test that saved eval run has required schema fields.

        Args:
            temp_runs_dir: Temporary runs directory.

        Returns:
            None.
        """
        from src.services.evaluation import EvaluationService
        from src.bootstrap import get_components
        from config import config
        
        components = get_components(config=config, force_mock=True, verbose=False)
        service = EvaluationService(components, runs_dir=temp_runs_dir)
        
        result = service.run_evaluation(scenario="diary", save=True)
        
        # Load and verify schema
        run_file = temp_runs_dir / result.run_id / "eval_run.json"
        with open(run_file) as f:
            data = json.load(f)
        
        # Required fields per spec
        assert "run_id" in data
        assert "timestamp" in data
        assert "config" in data
        assert "dataset" in data
        assert "metrics" in data
        assert "warnings" in data
        
        # Config should have model info
        config_data = data["config"]
        assert "embedding_provider" in config_data
        assert "embedding_model" in config_data
        assert "llm_provider" in config_data
        assert "precision_k" in config_data
    
    def test_eval_runs_can_be_loaded(self, temp_runs_dir: Path) -> None:
        """Test that saved runs can be loaded back.

        Args:
            temp_runs_dir: Temporary runs directory.

        Returns:
            None.
        """
        from src.services.evaluation import EvaluationService
        from src.bootstrap import get_components
        from config import config
        
        components = get_components(config=config, force_mock=True, verbose=False)
        service = EvaluationService(components, runs_dir=temp_runs_dir)
        
        # Save a run
        result1 = service.run_evaluation(scenario="diary", save=True)
        
        # Load it back
        loaded = service.load_run(result1.run_id)
        
        assert loaded is not None
        assert loaded.run_id == result1.run_id
        assert loaded.dataset == result1.dataset
        assert loaded.success == result1.success
    
    def test_eval_compare_detects_differences(self, temp_runs_dir: Path) -> None:
        """Test that comparison detects config/metric differences.

        Args:
            temp_runs_dir: Temporary runs directory.

        Returns:
            None.
        """
        from src.services.evaluation import EvaluationService, EvalRunResult, EvalConfig
        import json
        from datetime import datetime
        
        # Create two mock runs with different configs
        run_a_dir = temp_runs_dir / "run_a"
        run_b_dir = temp_runs_dir / "run_b"
        run_a_dir.mkdir()
        run_b_dir.mkdir()
        
        run_a = {
            "run_id": "run_a",
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": "abc123",
            "config": {
                "embedding_provider": "local",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": 1024,
                "llm_provider": "ollama",
                "llm_model": "qwen2.5:7b",
                "precision_k": 5,
                "scenario": "diary",
                "using_mock_embeddings": False,
                "using_mock_llm": False,
            },
            "dataset": "diary",
            "metrics": {
                "retrieval": {"precision_at_k": 0.6, "recall": 0.5},
                "counts": {"episodes": 10},
            },
            "warnings": [],
            "duration_seconds": 10.0,
            "success": True,
        }
        
        run_b = {
            "run_id": "run_b",
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": "def456",
            "config": {
                "embedding_provider": "mock",
                "embedding_model": "mock",
                "embedding_dimension": 384,
                "llm_provider": "mock",
                "llm_model": "mock",
                "precision_k": 5,
                "scenario": "diary",
                "using_mock_embeddings": True,
                "using_mock_llm": True,
            },
            "dataset": "diary",
            "metrics": {
                "retrieval": {"precision_at_k": 0.2, "recall": 0.1},
                "counts": {"episodes": 10},
            },
            "warnings": ["MOCK_EMBEDDINGS"],
            "duration_seconds": 5.0,
            "success": True,
        }
        
        with open(run_a_dir / "eval_run.json", "w") as f:
            json.dump(run_a, f)
        with open(run_b_dir / "eval_run.json", "w") as f:
            json.dump(run_b, f)
        
        # Create service and compare
        from src.bootstrap import get_components
        from config import config
        
        components = get_components(config=config, force_mock=True, verbose=False)
        service = EvaluationService(components, runs_dir=temp_runs_dir)
        
        comparison = service.compare_runs("run_a", "run_b")
        
        assert comparison is not None
        assert len(comparison.config_diffs) > 0, "Should detect config differences"
        assert len(comparison.metric_diffs) > 0, "Should detect metric differences"
        
        # Should detect embedding provider difference
        assert "embedding_provider" in comparison.config_diffs
    
    def test_eval_list_returns_runs(self, temp_runs_dir: Path) -> None:
        """Test that list_runs returns available runs.

        Args:
            temp_runs_dir: Temporary runs directory.

        Returns:
            None.
        """
        from src.services.evaluation import EvaluationService
        from src.bootstrap import get_components
        from config import config
        
        components = get_components(config=config, force_mock=True, verbose=False)
        service = EvaluationService(components, runs_dir=temp_runs_dir)
        
        # Initially empty
        runs = service.list_runs()
        assert len(runs) == 0
        
        # Create a run
        service.run_evaluation(scenario="diary", save=True)
        
        # Now should have one
        runs = service.list_runs()
        assert len(runs) == 1


class TestDemoDataPolicy:
    """Test demo data safety policy compliance."""
    
    def test_demo_data_readme_exists(self) -> None:
        """Verify `demo_data/README.md` exists.

        Returns:
            None.
        """
        readme_path = PROJECT_ROOT / "demo_data" / "README.md"
        assert readme_path.exists(), "demo_data/README.md not found"
    
    def test_demo_data_readme_has_policy(self) -> None:
        """Verify README contains data policy information.

        Returns:
            None.
        """
        readme_path = PROJECT_ROOT / "demo_data" / "README.md"
        content = readme_path.read_text()
        
        # Should mention key policy points
        assert "sensitive" in content.lower(), "README should mention sensitive data"
        assert "synthetic" in content.lower() or "fictional" in content.lower(), \
            "README should mention synthetic/fictional data"
    
    def test_fixtures_contain_only_synthetic_data(self) -> None:
        """Verify fixtures don't contain obviously real data.

        Returns:
            None.
        """
        fixtures_path = PROJECT_ROOT / "demo_data" / "fixtures.json"
        
        with open(fixtures_path) as f:
            data = json.load(f)
        
        # Check all episodes
        for ep in data["episodes"]:
            text = ep["text"].lower()
            
            # Should not contain real API keys
            assert "sk-" not in text, "Fixture contains API key-like string"
            assert "api_key" not in text, "Fixture mentions API key"
            
            # Should not contain real email addresses (basic check)
            assert "@gmail.com" not in text, "Fixture contains real email"
            assert "@yahoo.com" not in text, "Fixture contains real email"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

