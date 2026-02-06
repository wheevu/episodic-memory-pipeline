"""
Evaluation service for the episodic memory pipeline.

This module contains business logic for running evaluations.
Returns plain dataclasses - no Rich/Typer imports.
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.bootstrap import PipelineComponents


@dataclass
class EvalConfig:
    """Configuration snapshot for an evaluation run."""

    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    llm_provider: str
    llm_model: str
    precision_k: int
    scenario: str
    using_mock_embeddings: bool
    using_mock_llm: bool


@dataclass
class EvalRunResult:
    """Result of an evaluation run."""

    run_id: str
    timestamp: str
    git_commit: Optional[str]
    config: EvalConfig
    dataset: str
    metrics: Dict[str, Any]
    warnings: List[str]
    duration_seconds: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            A JSON-serializable dictionary representation of this run.
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "config": asdict(self.config),
            "dataset": self.dataset,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class EvalComparison:
    """Comparison between two evaluation runs."""

    run_a_id: str
    run_b_id: str
    config_diffs: Dict[str, tuple]  # key -> (value_a, value_b)
    metric_diffs: Dict[str, tuple]  # key -> (value_a, value_b, delta)
    warnings: List[str]


def get_git_commit() -> Optional[str]:
    """Get current git commit hash, if available.

    Returns:
        Short git commit hash string if available; otherwise None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp.

    Returns:
        A run identifier string suitable for directory naming.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class EvaluationService:
    """
    Service for running and managing evaluations.

    This service wraps the EvaluationRunner and provides functionality
    for storing, loading, and comparing evaluation runs.
    """

    def __init__(self, components: "PipelineComponents", runs_dir: Optional[Path] = None) -> None:
        """
        Initialize the evaluation service.

        Args:
            components: Pipeline components from bootstrap
            runs_dir: Directory for storing run outputs (default: runs/eval/)

        Returns:
            None.
        """
        self.components = components
        self.runs_dir = runs_dir or Path("runs/eval")
        self._runner = None

    def _get_runner(self, precision_k: int = 5) -> Any:
        """Create an evaluation runner.

        Args:
            precision_k: K value for precision@k metric.

        Returns:
            An `EvaluationRunner` instance from the bootstrap components.
        """
        return self.components.EvaluationRunner(
            embedding_provider=self.components.embedding_provider,
            llm=self.components.llm,
            precision_k=precision_k,
        )

    def _get_config(self, scenario: str, precision_k: int) -> EvalConfig:
        """Build config snapshot from current components.

        Args:
            scenario: Scenario name to record.
            precision_k: K used for precision@k metric.

        Returns:
            An `EvalConfig` snapshot describing the run configuration.
        """
        from config import config

        emb = self.components.embedding_provider
        llm = self.components.llm

        return EvalConfig(
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_dimension=config.embedding_dimension,
            llm_provider=config.llm_provider,
            llm_model=(
                config.ollama_model if config.llm_provider == "ollama" else config.llm_model
            ),
            precision_k=precision_k,
            scenario=scenario,
            using_mock_embeddings=getattr(emb, "is_mock", False),
            using_mock_llm=getattr(llm, "is_mock", False),
        )

    def run_evaluation(
        self,
        scenario: str = "diary",
        precision_k: int = 5,
        save: bool = True,
        out_dir: Optional[Path] = None,
    ) -> EvalRunResult:
        """
        Run an evaluation scenario and optionally save results.

        Args:
            scenario: Name of the evaluation scenario
            precision_k: K value for precision@k metric
            save: Whether to save results to disk
            out_dir: Custom output directory (default: runs/eval/<run_id>/)

        Returns:
            EvalRunResult with all metrics and metadata
        """
        run_id = generate_run_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        git_commit = get_git_commit()
        config = self._get_config(scenario, precision_k)

        warnings = []
        if config.using_mock_embeddings:
            warnings.append("MOCK_EMBEDDINGS: Retrieval metrics are not meaningful")
        if config.using_mock_llm:
            warnings.append("MOCK_LLM: Fact extraction metrics may not be meaningful")

        # Get scenario
        try:
            eval_scenario = self.components.get_scenario(scenario)
        except ValueError as e:
            return EvalRunResult(
                run_id=run_id,
                timestamp=timestamp,
                git_commit=git_commit,
                config=config,
                dataset=scenario,
                metrics={},
                warnings=warnings,
                duration_seconds=0,
                success=False,
                error=str(e),
            )

        # Run evaluation
        runner = self._get_runner(precision_k)
        scenario_result = runner.run_scenario(eval_scenario)

        if not scenario_result.success:
            return EvalRunResult(
                run_id=run_id,
                timestamp=timestamp,
                git_commit=git_commit,
                config=config,
                dataset=scenario,
                metrics={},
                warnings=warnings,
                duration_seconds=scenario_result.duration_seconds,
                success=False,
                error=scenario_result.error,
            )

        # Build metrics dict
        metrics = scenario_result.metrics.to_dict()

        result = EvalRunResult(
            run_id=run_id,
            timestamp=timestamp,
            git_commit=git_commit,
            config=config,
            dataset=scenario,
            metrics=metrics,
            warnings=warnings,
            duration_seconds=scenario_result.duration_seconds,
            success=True,
        )

        # Save if requested
        if save:
            self.save_run(result, out_dir)

        return result

    def save_run(self, result: EvalRunResult, out_dir: Optional[Path] = None) -> Path:
        """
        Save an evaluation run to disk.

        Args:
            result: The evaluation result to save
            out_dir: Custom output directory

        Returns:
            Path to the saved run directory
        """
        if out_dir is None:
            out_dir = self.runs_dir / result.run_id

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save eval_run.json
        run_file = out_dir / "eval_run.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return out_dir

    def load_run(self, run_id_or_path: str) -> Optional[EvalRunResult]:
        """
        Load an evaluation run from disk.

        Args:
            run_id_or_path: Either a run ID (looks in runs_dir) or a path

        Returns:
            EvalRunResult if found, None otherwise
        """
        # Check if it's a path
        path = Path(run_id_or_path)
        if not path.exists():
            # Try as run_id
            path = self.runs_dir / run_id_or_path

        run_file = path / "eval_run.json" if path.is_dir() else path

        if not run_file.exists():
            return None

        with open(run_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvalRunResult(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            git_commit=data.get("git_commit"),
            config=EvalConfig(**data["config"]),
            dataset=data["dataset"],
            metrics=data["metrics"],
            warnings=data.get("warnings", []),
            duration_seconds=data["duration_seconds"],
            success=data["success"],
            error=data.get("error"),
        )

    def list_runs(self) -> List[str]:
        """List all available run IDs.

        Returns:
            A list of run directory names in reverse chronological order.
        """
        if not self.runs_dir.exists():
            return []

        runs = []
        for path in sorted(self.runs_dir.iterdir(), reverse=True):
            if path.is_dir() and (path / "eval_run.json").exists():
                runs.append(path.name)
        return runs

    def compare_runs(self, run_a: str, run_b: str) -> Optional[EvalComparison]:
        """
        Compare two evaluation runs.

        Args:
            run_a: First run ID or path
            run_b: Second run ID or path

        Returns:
            EvalComparison with diffs, or None if runs not found
        """
        result_a = self.load_run(run_a)
        result_b = self.load_run(run_b)

        if result_a is None or result_b is None:
            return None

        config_diffs = {}
        warnings = []

        # Compare configs
        config_a = asdict(result_a.config)
        config_b = asdict(result_b.config)

        for key in set(config_a.keys()) | set(config_b.keys()):
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                config_diffs[key] = (val_a, val_b)

        # Compare metrics
        metric_diffs = {}

        def extract_metrics(m: dict, prefix: str = "") -> dict:
            """Flatten nested metrics dict."""
            flat = {}
            for k, v in m.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    flat.update(extract_metrics(v, f"{key}."))
                elif isinstance(v, (int, float)):
                    flat[key] = v
            return flat

        metrics_a = extract_metrics(result_a.metrics)
        metrics_b = extract_metrics(result_b.metrics)

        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        for key in sorted(all_keys):
            val_a = metrics_a.get(key, 0)
            val_b = metrics_b.get(key, 0)
            if val_a != val_b:
                delta = val_b - val_a if isinstance(val_a, (int, float)) else None
                metric_diffs[key] = (val_a, val_b, delta)

        # Add warnings for significant differences
        if config_diffs.get("using_mock_embeddings"):
            warnings.append("Mock embedding status differs between runs")
        if config_diffs.get("using_mock_llm"):
            warnings.append("Mock LLM status differs between runs")
        if config_diffs.get("embedding_model"):
            warnings.append("Embedding model differs between runs")

        return EvalComparison(
            run_a_id=result_a.run_id,
            run_b_id=result_b.run_id,
            config_diffs=config_diffs,
            metric_diffs=metric_diffs,
            warnings=warnings,
        )
