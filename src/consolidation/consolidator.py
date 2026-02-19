"""
Consolidation pipeline - orchestrates periodic memory consolidation.

Consolidation transforms raw episodic memories into:
1. Topic-level narrative summaries
2. Stable semantic facts

This mimics how human memory consolidates during sleep.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from ..models import Summary
from ..storage import LanceStore
from .fact_extractor import FactExtractor
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Represents the outcome and statistics of a consolidation run.

    Attributes:
        run_id: Unique identifier for the consolidation run.
        topic: Topic consolidated (or None if not topic-specific).
        episodes_processed: Number of episodes processed in this run.
        summaries_created: Number of summaries created.
        facts_extracted: Number of new facts created.
        facts_updated: Number of facts updated (superseded + replaced).
        facts_contradicted: Number of facts marked contradicted.
        duration_seconds: Total runtime for the consolidation run.
    """

    run_id: str
    topic: Optional[str]
    episodes_processed: int
    summaries_created: int
    facts_extracted: int
    facts_updated: int
    facts_contradicted: int
    duration_seconds: float


class ConsolidationPipeline:
    """
    Orchestrates memory consolidation.

    When to run consolidation:
    1. Periodically (e.g., daily, triggered by scheduler)
    2. When episode count exceeds threshold
    3. Manually triggered by user

    How conflicts are handled:
    - New information updates existing facts with boosted confidence
    - Contradictions mark old facts as superseded
    - Both old and new versions are preserved for provenance

    How older detail is preserved:
    - Episodes are never deleted, only marked as consolidated
    - Summaries link back to source episodes
    - Facts link to all supporting episodes
    - Hierarchical summaries aggregate over longer periods
    """

    def __init__(
        self,
        lance_store: LanceStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
        episode_threshold: int = 5,
        age_threshold_days: int = 7,
    ) -> None:
        """Initialize the consolidation pipeline.

        Args:
            lance_store: Unified metadata and vector store.
            embedding_provider: Provider used to embed summaries and facts.
            llm: Provider used for summarization and fact extraction.
            episode_threshold: Minimum unconsolidated episode count to trigger consolidation.
            age_threshold_days: Maximum days since last consolidation before triggering.
        """
        self.lance_store = lance_store
        self.embedding_provider = embedding_provider
        self.summarizer = Summarizer(llm)
        self.fact_extractor = FactExtractor(llm)
        self.episode_threshold = episode_threshold
        self.age_threshold_days = age_threshold_days

    def consolidate_topic(self, topic: str) -> ConsolidationResult:
        """Consolidate a single topic by summarizing episodes and extracting facts.

        Args:
            topic: Topic to consolidate.

        Returns:
            A `ConsolidationResult` with run statistics.
        """
        run_id = str(uuid4())
        start_time = datetime.now(timezone.utc)

        # Get unconsolidated episodes for this topic
        episodes = self.lance_store.get_unconsolidated_episodes(topic=topic)

        if not episodes:
            return ConsolidationResult(
                run_id=run_id,
                topic=topic,
                episodes_processed=0,
                summaries_created=0,
                facts_extracted=0,
                facts_updated=0,
                facts_contradicted=0,
                duration_seconds=0.0,
            )

        # Get existing facts for comparison
        existing_facts = self.lance_store.get_facts(topic=topic)

        # Generate summary
        summary_result = self.summarizer.summarize(episodes, topic)
        summary = summary_result.summary

        # Embed and store summary
        summary_embedding = self.embedding_provider.embed_text(summary.to_embedding_text())
        self.lance_store.save_summary(
            summary, summary_embedding, source_episode_ids=[ep.id for ep in episodes]
        )

        # Extract facts
        fact_result = self.fact_extractor.extract_facts(episodes, topic, existing_facts)

        # Store new facts
        for fact in fact_result.new_facts:
            fact_embedding = self.embedding_provider.embed_text(fact.to_embedding_text())
            self.lance_store.save_fact(
                fact, fact_embedding, source_episode_ids=fact_result.source_episode_ids
            )

        # Handle updated facts
        for old_id, new_fact in fact_result.updated_facts:
            fact_embedding = self.embedding_provider.embed_text(new_fact.to_embedding_text())
            self.lance_store.save_fact(
                new_fact,
                fact_embedding,
                source_episode_ids=fact_result.source_episode_ids,
            )
            self.lance_store.supersede_fact(old_id, new_fact)

        # Handle contradicted facts (deactivate so they no longer appear in retrieval)
        for fact_id in fact_result.contradicted_fact_ids:
            self.lance_store.set_fact_active(fact_id, False)
            logger.info("Deactivated contradicted fact: %s", fact_id[:8])

        # Mark episodes as consolidated
        self.lance_store.mark_episodes_consolidated([ep.id for ep in episodes])

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return ConsolidationResult(
            run_id=run_id,
            topic=topic,
            episodes_processed=len(episodes),
            summaries_created=1,
            facts_extracted=len(fact_result.new_facts),
            facts_updated=len(fact_result.updated_facts),
            facts_contradicted=len(fact_result.contradicted_fact_ids),
            duration_seconds=duration,
        )

    def consolidate_all(self) -> list[ConsolidationResult]:
        """Consolidate all topics that meet consolidation criteria.

        Returns:
            A list of `ConsolidationResult`, one per consolidated topic.
        """
        topics = self.lance_store.get_topics_needing_consolidation(
            min_episodes=self.episode_threshold, max_age_days=self.age_threshold_days
        )

        results = []
        for topic in topics:
            result = self.consolidate_topic(topic)
            results.append(result)

        return results

    def should_consolidate(self, topic: Optional[str] = None) -> bool:
        """Return True if consolidation is needed for a topic (or any topic).

        Args:
            topic: Specific topic to check, or None to check whether any topic qualifies.

        Returns:
            True if consolidation is needed.
        """
        if topic:
            episodes = self.lance_store.get_unconsolidated_episodes(topic=topic)
            return len(episodes) >= self.episode_threshold
        else:
            topics = self.lance_store.get_topics_needing_consolidation(
                min_episodes=self.episode_threshold, max_age_days=self.age_threshold_days
            )
            return len(topics) > 0

    def create_weekly_summary(self, topic: str) -> Optional[Summary]:
        """Create a weekly summary from recent episodes for a topic.

        Args:
            topic: Topic to summarize.

        Returns:
            A `Summary` if there are recent episodes; otherwise None.
        """
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        episodes = self.lance_store.get_episodes(topic=topic, since=week_ago, limit=100)

        if not episodes:
            return None

        result = self.summarizer.summarize(episodes, topic)
        result.summary.summary_level = 1

        return result.summary

    def create_monthly_summary(self, topic: str) -> Optional[Summary]:
        """Create a monthly summary by aggregating weekly summaries.

        Args:
            topic: Topic to summarize.

        Returns:
            A higher-level `Summary` if enough weekly summaries exist; otherwise None.
        """
        month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        weekly_summaries = self.lance_store.get_summaries(topic=topic, level=1, since=month_ago)

        if len(weekly_summaries) < 2:
            return None

        return self.summarizer.create_higher_level_summary(weekly_summaries, topic, level=2)
