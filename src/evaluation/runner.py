"""
Evaluation runner for executing evaluation scenarios.

Provides the framework for running end-to-end evaluations of the memory pipeline.
"""

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..consolidation import ConsolidationPipeline
from ..embeddings import EmbeddingProvider
from ..ingestion import IngestionPipeline
from ..llm import LLMProvider
from ..retrieval import RetrievalEngine
from ..storage import LanceStore
from .metrics import (
    ConsolidationCompressionMetric,
    EvaluationMetrics,
    FactConflictRateMetric,
    RetrievalPrecisionMetric,
)


@dataclass
class EvaluationQuery:
    """A query with expected results for evaluation."""

    query_text: str
    expected_episode_ids: list[str] = field(default_factory=list)
    expected_topics: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of running an evaluation scenario."""

    scenario_name: str
    success: bool
    metrics: EvaluationMetrics
    error: Optional[str] = None
    duration_seconds: float = 0.0


class EvaluationScenario(ABC):
    """Abstract base class for evaluation scenarios."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return scenario name.

        Returns:
            A short identifier for the scenario.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return scenario description.

        Returns:
            A human-readable description of the scenario.
        """
        pass

    @abstractmethod
    def get_episodes(self) -> list[tuple[str, list[str]]]:
        """
        Return episodes for ingestion.

        Returns:
            List of (text, expected_topics) tuples
        """
        pass

    @abstractmethod
    def get_queries(self) -> list[EvaluationQuery]:
        """
        Return queries for retrieval evaluation.

        Returns:
            List of EvaluationQuery objects
        """
        pass


class DiaryScenario(EvaluationScenario):
    """
    Diary evaluation scenario.

    Simulates a personal diary with daily entries over a week,
    covering multiple topics and tracking state changes.
    """

    @property
    def name(self) -> str:
        """Return the short scenario identifier.

        Returns:
            The scenario identifier string.
        """
        return "diary"

    @property
    def description(self) -> str:
        """Return a human-readable description of the scenario.

        Returns:
            A human-readable description string.
        """
        return "Personal diary entries over one week with learning and work topics"

    def get_episodes(self) -> list[tuple[str, list[str]]]:
        """Return diary entries with expected topics.

        Returns:
            A list of `(text, expected_topics)` tuples used for ingestion.
        """
        return [
            # Day 1 - Monday
            (
                "Started a new project at work today. The client wants a recommendation system. "
                "I'm excited but also nervous about the timeline - we only have 6 weeks.",
                ["work", "project"],
            ),
            (
                "Decided to learn Korean this year. My friend recommended Duolingo to start. "
                "Downloaded the app and completed the first lesson - basic greetings!",
                ["learning", "korean"],
            ),
            # Day 2 - Tuesday
            (
                "Had a productive meeting with the team about the recommendation system. "
                "We decided to use collaborative filtering as the main approach.",
                ["work", "project"],
            ),
            (
                "Korean practice day 2. Learned numbers 1-10: 하나, 둘, 셋, 넷, 다섯... "
                "It's harder than I expected but I'm enjoying it.",
                ["learning", "korean"],
            ),
            # Day 3 - Wednesday
            (
                "The client changed requirements again. Now they want content-based filtering too. "
                "This means more work but it's actually a better approach.",
                ["work", "project"],
            ),
            # Day 4 - Thursday
            (
                "Practiced Korean for 30 minutes. Can now introduce myself: 안녕하세요, 제 이름은... "
                "My pronunciation still needs work.",
                ["learning", "korean"],
            ),
            (
                "I prefer working from home on Thursdays. The office is too noisy for deep work.",
                ["work", "preference"],
            ),
            # Day 5 - Friday
            (
                "Finished the initial prototype of the recommendation engine. It's rough but it works! "
                "The team is happy with the progress.",
                ["work", "project"],
            ),
            (
                "Joined a Korean language exchange group online. Met someone from Seoul who wants "
                "to practice English. We'll do weekly video calls.",
                ["learning", "korean", "social"],
            ),
            # Day 6 - Saturday
            (
                "Took a break from coding. Spent the day reading a book about machine learning. "
                "Found some ideas that might help with the recommendation system.",
                ["learning", "work"],
            ),
            # Day 7 - Sunday
            (
                "Weekly review: Made good progress on both the work project and Korean learning. "
                "Goals for next week: finish the recommendation system MVP and learn 50 new Korean words.",
                ["reflection", "work", "learning"],
            ),
        ]

    def get_queries(self) -> list[EvaluationQuery]:
        """Return evaluation queries with expected results.

        Returns:
            A list of `EvaluationQuery` objects used to evaluate retrieval.
        """
        return [
            EvaluationQuery(
                query_text="What am I learning?",
                expected_topics=["learning", "korean"],
            ),
            EvaluationQuery(
                query_text="Tell me about my work project",
                expected_topics=["work", "project"],
            ),
            EvaluationQuery(
                query_text="What are my preferences?",
                expected_topics=["preference"],
            ),
            EvaluationQuery(
                query_text="How is my Korean learning going?",
                expected_topics=["korean", "learning"],
            ),
            EvaluationQuery(
                query_text="What happened this week?",
                expected_topics=["reflection"],
            ),
        ]


class EvaluationRunner:
    """
    Runs evaluation scenarios against the memory pipeline.

    Creates an isolated environment for each evaluation to ensure
    reproducible results.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
        precision_k: int = 5,
    ) -> None:
        """
        Initialize evaluation runner.

        Args:
            embedding_provider: Embedding provider to use
            llm: LLM provider to use
            precision_k: K value for precision@k metric

        Returns:
            None.
        """
        self.embedding_provider = embedding_provider
        self.llm = llm
        self.precision_k = precision_k

        # Initialize metrics
        self.precision_metric = RetrievalPrecisionMetric(k=precision_k)
        self.conflict_metric = FactConflictRateMetric()
        self.compression_metric = ConsolidationCompressionMetric()

    def run_scenario(
        self,
        scenario: EvaluationScenario,
        cleanup: bool = True,
    ) -> ScenarioResult:
        """
        Run a single evaluation scenario.

        Args:
            scenario: The scenario to run
            cleanup: Whether to clean up temp files after

        Returns:
            ScenarioResult with metrics
        """
        start_time = datetime.now(timezone.utc)
        temp_dir = None

        try:
            # Create isolated environment
            temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_{scenario.name}_"))

            lance_store = LanceStore(
                temp_dir / "lancedb", embedding_dimension=self.embedding_provider.dimension
            )

            # Initialize pipelines
            ingestion = IngestionPipeline(
                lance_store,
                self.embedding_provider,
                self.llm,
                worthiness_threshold=0.3,  # Lower threshold for eval
            )

            consolidation = ConsolidationPipeline(
                lance_store,
                self.embedding_provider,
                self.llm,
                episode_threshold=3,  # Consolidate earlier for eval
            )

            retrieval = RetrievalEngine(lance_store, self.embedding_provider, self.llm)

            # Step 1: Ingest episodes
            ingested_episodes = []
            episode_topic_map = {}  # Map episode_id -> expected_topics

            for text, expected_topics in scenario.get_episodes():
                result = ingestion.ingest(text, source="eval", force=True)
                if result.success and result.episode:
                    ingested_episodes.append(result.episode)
                    episode_topic_map[result.episode.id] = expected_topics

            # Step 2: Run consolidation
            consolidation.consolidate_all()

            # Step 3: Run retrieval queries and collect results
            query_results = []
            for query in scenario.get_queries():
                result = retrieval.query(query.query_text, synthesize=False)

                # Map retrieved episodes to their IDs
                retrieved_ids = [ep.id for ep in result.episodes]

                # Find expected episode IDs based on expected topics
                expected_ids = [
                    ep_id
                    for ep_id, topics in episode_topic_map.items()
                    if any(t in query.expected_topics for t in topics)
                ]

                query_results.append((retrieved_ids, expected_ids))

            # Step 4: Compute metrics
            precision_result = self.precision_metric.evaluate_multiple(query_results)

            all_facts = lance_store.get_facts()
            conflict_result = self.conflict_metric.evaluate(all_facts)

            all_episodes = lance_store.get_episodes()
            all_summaries = lance_store.get_summaries()
            compression_result = self.compression_metric.evaluate(all_episodes, all_summaries)

            # Aggregate metrics
            metrics = EvaluationMetrics(
                retrieval_precision=precision_result,
                fact_conflict=conflict_result,
                compression=compression_result,
                scenario_name=scenario.name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                episode_count=len(all_episodes),
                fact_count=len(all_facts),
                summary_count=len(all_summaries),
                using_mock_embeddings=getattr(self.embedding_provider, "is_mock", False),
                using_mock_llm=getattr(self.llm, "is_mock", False),
            )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            return ScenarioResult(
                scenario_name=scenario.name,
                success=True,
                metrics=metrics,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return ScenarioResult(
                scenario_name=scenario.name,
                success=False,
                metrics=EvaluationMetrics(scenario_name=scenario.name),
                error=str(e),
                duration_seconds=duration,
            )
        finally:
            if cleanup and temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def run_all_scenarios(
        self,
        scenarios: list[EvaluationScenario] = None,
    ) -> list[ScenarioResult]:
        """
        Run multiple evaluation scenarios.

        Args:
            scenarios: List of scenarios (default: [DiaryScenario()])

        Returns:
            List of ScenarioResult objects
        """
        if scenarios is None:
            scenarios = [DiaryScenario()]

        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

        return results


def get_scenario(name: str) -> EvaluationScenario:
    """
    Factory function to get evaluation scenario by name.

    Args:
        name: Scenario name ("diary", etc.)

    Returns:
        EvaluationScenario instance

    Raises:
        ValueError: If scenario name is unknown
    """
    scenarios = {
        "diary": DiaryScenario,
    }

    if name not in scenarios:
        available = ", ".join(scenarios.keys())
        raise ValueError(f"Unknown scenario: {name}. Available: {available}")

    return scenarios[name]()
