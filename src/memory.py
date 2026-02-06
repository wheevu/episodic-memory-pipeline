"""
MemorySystem — single-entry-point facade for agent integration.

Usage::

    from src.memory import MemorySystem

    mem = MemorySystem()                       # uses default config / env vars
    mem = MemorySystem(force_mock=True)        # mock providers for testing

    # Store a memory
    result = mem.remember("I started learning Korean today")

    # Recall by natural-language query (with optional LLM synthesis)
    result = mem.recall("What am I learning?")

    # Recall narrative / journey for a topic
    result = mem.recall_narrative("korean")

    # Quick fact lookup (no LLM synthesis)
    facts = mem.quick_lookup("What language am I learning?")

    # Get compact topic context (for stuffing into an agent prompt)
    ctx = mem.get_context("korean")

    # Run consolidation (episodes → summaries + facts)
    results = mem.consolidate()

    # Soft-delete a memory
    mem.forget(episode_id="abc123")
    mem.forget(fact_id="def456")

    # Inspect system state
    stats = mem.stats()
    topics = mem.topics()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import Config
    from src.consolidation.consolidator import ConsolidationResult
    from src.ingestion.pipeline import IngestionResult
    from src.models import Episode, Fact
    from src.retrieval.engine import QueryResult

logger = logging.getLogger(__name__)


class MemorySystem:
    """High-level facade over the episodic-memory pipeline.

    Encapsulates bootstrap wiring, import ordering (FAISS/SentenceTransformers),
    and pipeline construction so that agent code only needs a single object.

    All heavy initialisation (model loading, DB/vector-store creation) happens
    lazily on first use, **not** at construction time, unless ``eager=True``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional["Config"] = None,
        force_mock: bool = False,
        verbose: bool = False,
        eager: bool = False,
    ) -> None:
        """Create a MemorySystem.

        Args:
            config: Pipeline configuration.  Uses the global ``Config()``
                singleton (which reads env vars / ``.env``) when *None*.
            force_mock: If *True*, use mock embedding + LLM providers
                regardless of config.  Useful for unit tests.
            verbose: If *True*, print provider-selection messages during
                bootstrap.
            eager: If *True*, bootstrap immediately instead of on first
                access.  Handy when you want startup errors to surface
                early.
        """
        self._config = config
        self._force_mock = force_mock
        self._verbose = verbose

        # Lazily populated by _ensure_components()
        self._components = None
        self._ingestion = None
        self._consolidation = None
        self._retrieval = None

        if eager:
            self._ensure_components()

    # ------------------------------------------------------------------
    # Lazy wiring (respects FAISS import ordering via bootstrap)
    # ------------------------------------------------------------------

    def _ensure_components(self) -> None:
        """Bootstrap pipeline components on first access."""
        if self._components is not None:
            return

        from src.bootstrap import get_components

        self._components = get_components(
            config=self._config,
            force_mock=self._force_mock,
            verbose=self._verbose,
        )
        logger.debug("MemorySystem: components bootstrapped")

    @property
    def _ingestion_pipeline(self):
        """Lazily create the ingestion pipeline."""
        if self._ingestion is None:
            self._ensure_components()
            c = self._components
            # Resolve threshold from config (fall back to bootstrap config)
            from config import config as default_config

            cfg = self._config or default_config
            self._ingestion = c.IngestionPipeline(
                c.database,
                c.vector_store,
                c.embedding_provider,
                c.llm,
                worthiness_threshold=cfg.memory_worthiness_threshold,
            )
        return self._ingestion

    @property
    def _consolidation_pipeline(self):
        """Lazily create the consolidation pipeline."""
        if self._consolidation is None:
            self._ensure_components()
            c = self._components
            from config import config as default_config

            cfg = self._config or default_config
            self._consolidation = c.ConsolidationPipeline(
                c.database,
                c.vector_store,
                c.embedding_provider,
                c.llm,
                episode_threshold=cfg.consolidation_episode_threshold,
                age_threshold_days=cfg.consolidation_age_days,
            )
        return self._consolidation

    @property
    def _retrieval_engine(self):
        """Lazily create the retrieval engine."""
        if self._retrieval is None:
            self._ensure_components()
            c = self._components
            self._retrieval = c.RetrievalEngine(
                c.database,
                c.vector_store,
                c.embedding_provider,
                c.llm,
            )
        return self._retrieval

    @property
    def database(self):
        """Direct access to the ``Database`` instance (escape hatch).

        Returns:
            The active ``Database`` instance.
        """
        self._ensure_components()
        return self._components.database

    @property
    def vector_store(self):
        """Direct access to the ``VectorStore`` instance (escape hatch).

        Returns:
            The active ``VectorStore`` instance.
        """
        self._ensure_components()
        return self._components.vector_store

    # ------------------------------------------------------------------
    # remember — store a new memory
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        source: str = "agent",
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        force: bool = False,
    ) -> "IngestionResult":
        """Ingest text into the memory system.

        The pipeline classifies the text for memory-worthiness, extracts
        a structured episode, generates an embedding, and persists both
        the database record and the vector index entry.

        Args:
            text: Raw input text to store.
            source: Source label (e.g. ``"agent"``, ``"chat"``, ``"note"``).
            session_id: Optional session grouping key.
            timestamp: When the event occurred (defaults to *now* UTC).
            force: If *True*, skip the worthiness classifier and always store.

        Returns:
            An ``IngestionResult`` describing what happened.
        """
        return self._ingestion_pipeline.ingest(
            text,
            source=source,
            session_id=session_id,
            timestamp=timestamp,
            force=force,
        )

    def remember_batch(
        self,
        texts: list[str],
        *,
        source: str = "agent",
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        force: bool = False,
    ) -> list["IngestionResult"]:
        """Ingest multiple texts.

        Processes sequentially; vector store is persisted once at the end.

        Args:
            texts: Input texts to store.
            source: Source label applied to all items.
            session_id: Optional session grouping key.
            timestamp: Optional shared timestamp.
            force: If *True*, skip worthiness checks for all items.

        Returns:
            A list of ``IngestionResult``, one per input text.
        """
        return self._ingestion_pipeline.ingest_batch(
            texts,
            source=source,
            session_id=session_id,
            timestamp=timestamp,
            force=force,
        )

    # ------------------------------------------------------------------
    # recall — query the memory system
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        synthesize: bool = True,
    ) -> "QueryResult":
        """Query the memory system with natural language.

        The engine analyses the query (via LLM) to choose a retrieval
        strategy (semantic vs narrative), fetches relevant memories, and
        optionally synthesises an answer.

        Args:
            query: Natural-language question or search phrase.
            synthesize: If *True* (default), generate an LLM-synthesised
                answer from the retrieved context.  Set to *False* for
                retrieval-only mode.

        Returns:
            A ``QueryResult`` with the answer text, supporting episodes/
            facts/summaries, and confidence.
        """
        return self._retrieval_engine.query(query, synthesize=synthesize)

    def recall_narrative(
        self,
        topic_or_query: str,
        *,
        is_topic: bool = False,
    ) -> "QueryResult":
        """Recall a narrative / journey about a topic.

        Optimised for "Tell me about …" style requests.

        Args:
            topic_or_query: Either an explicit topic name (when
                ``is_topic=True``) or a free-form query.
            is_topic: Treat the first argument as a known topic name
                rather than a query to analyse.

        Returns:
            A ``QueryResult`` with a narrative synthesis and supporting
            context.
        """
        return self._retrieval_engine.recall_narrative(topic_or_query, is_topic=is_topic)

    def quick_lookup(self, query: str) -> list["Fact"]:
        """Fast fact-only lookup without LLM synthesis.

        Args:
            query: Fact-oriented query (e.g. "What is my favourite food?").

        Returns:
            A list of matching ``Fact`` objects.
        """
        return self._retrieval_engine.quick_lookup(query)

    def get_context(self, topic: str, *, max_items: int = 5) -> dict:
        """Return a compact context bundle for a topic.

        Designed to be stuffed directly into an agent's system prompt.
        The returned dict contains recent episodes, facts, and the latest
        summary for the topic.

        Args:
            topic: Topic to fetch context for.
            max_items: Maximum episodes/facts to include.

        Returns:
            A dictionary with keys ``topic``, ``recent_episodes``,
            ``facts``, ``summary``, ``episode_count``, ``fact_count``.
        """
        return self._retrieval_engine.get_context(topic, max_items=max_items)

    # ------------------------------------------------------------------
    # consolidate — episodes → summaries + facts
    # ------------------------------------------------------------------

    def consolidate(
        self,
        topic: Optional[str] = None,
    ) -> list["ConsolidationResult"]:
        """Run memory consolidation.

        Transforms unconsolidated episodes into topic summaries and
        extracted facts — analogous to how human memory consolidates
        during sleep.

        Args:
            topic: If given, consolidate only this topic.  Otherwise,
                consolidate all topics that meet the threshold.

        Returns:
            A list of ``ConsolidationResult`` (one per topic processed).
        """
        if topic:
            result = self._consolidation_pipeline.consolidate_topic(topic)
            return [result]
        return self._consolidation_pipeline.consolidate_all()

    # ------------------------------------------------------------------
    # forget — soft-delete memories
    # ------------------------------------------------------------------

    def forget(
        self,
        *,
        episode_id: Optional[str] = None,
        fact_id: Optional[str] = None,
    ) -> bool:
        """Soft-delete a memory by deactivating it.

        Deactivated records are excluded from all retrieval queries.
        The underlying DB rows and (orphaned) vector entries are
        preserved for auditability; they simply stop appearing in
        results.

        Exactly one of ``episode_id`` or ``fact_id`` must be provided.

        Args:
            episode_id: Episode to deactivate.
            fact_id: Fact to deactivate.

        Returns:
            *True* if a record was deactivated; *False* if the record
            was not found.

        Raises:
            ValueError: If neither or both identifiers are provided.
        """
        if (episode_id is None) == (fact_id is None):
            raise ValueError("Provide exactly one of episode_id or fact_id")

        self._ensure_components()
        db = self._components.database

        if episode_id is not None:
            episode = db.get_episode(episode_id)
            if episode is None:
                logger.warning("forget: episode %s not found", episode_id)
                return False
            db.set_episode_active(episode_id, False)
            logger.info("Deactivated episode %s", episode_id[:8])
            return True

        # fact_id is not None (guaranteed by the guard above)
        fact = db.get_fact(fact_id)
        if fact is None:
            logger.warning("forget: fact %s not found", fact_id)
            return False
        db.set_fact_active(fact_id, False)
        logger.info("Deactivated fact %s", fact_id[:8])
        return True

    # ------------------------------------------------------------------
    # stats / topics — system introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return database and vector-store statistics.

        Returns:
            A dictionary with ``database`` and ``vector_store`` sub-dicts.
        """
        return self._ingestion_pipeline.get_statistics()

    def topics(self) -> list[dict]:
        """List all known topics with episode/fact counts.

        Returns:
            A list of topic info dicts (``name``, ``episode_count``, etc.).
        """
        self._ensure_components()
        return self._components.database.get_topics()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "initialised" if self._components is not None else "lazy"
        mock = " mock=True" if self._force_mock else ""
        return f"<MemorySystem {status}{mock}>"
