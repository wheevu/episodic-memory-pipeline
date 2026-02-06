"""
Semantic retrieval - vector similarity based memory lookup.

For queries like "What am I learning right now?" or "What do I know about X?"
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from ..embeddings import EmbeddingProvider
from ..models import Episode, Fact, Summary
from ..storage import Database, VectorStore


@dataclass
class SemanticResult:
    """Represents results of a semantic retrieval query.

    Attributes:
        episodes: Retrieved episodes (typically ranked by similarity).
        facts: Retrieved facts (typically ranked by similarity).
        summaries: Retrieved summaries (typically ranked by similarity).
        query_embedding_time: Time spent embedding the query, in seconds.
        search_time: Time spent performing vector search and DB hydration, in seconds.
    """

    episodes: list[Episode]
    facts: list[Fact]
    summaries: list[Summary]
    query_embedding_time: float
    search_time: float


class SemanticRetriever:
    """
    Semantic retrieval using vector similarity.

    Combines results from episodes, facts, and summaries
    with optional metadata filtering.
    """

    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize the semantic retriever.

        Args:
            database: Structured storage used to fetch full records by ID.
            vector_store: Vector index used for similarity search.
            embedding_provider: Provider used to embed the query text.
            top_k: Default number of results to retrieve per memory type.
            similarity_threshold: Minimum cosine similarity score to include.
        """
        self.database = database
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.threshold = similarity_threshold

    def search(
        self,
        query: str,
        search_episodes: bool = True,
        search_facts: bool = True,
        search_summaries: bool = True,
        topic_filter: Optional[str] = None,
        time_filter_since: Optional[datetime] = None,
        time_filter_until: Optional[datetime] = None,
        top_k: Optional[int] = None,
    ) -> SemanticResult:
        """Perform semantic similarity search across episodes, facts, and summaries.

        Args:
            query: Natural language query to embed and search for.
            search_episodes: If True, include episodes in the search.
            search_facts: If True, include facts in the search.
            search_summaries: If True, include summaries in the search.
            topic_filter: Optional topic constraint applied during retrieval.
            time_filter_since: Optional lower-bound time filter for episodes.
            time_filter_until: Optional upper-bound time filter for episodes.
            top_k: Optional override for the default `top_k`.

        Returns:
            A `SemanticResult` containing retrieved items and timing metadata.
        """
        import time

        k = top_k or self.top_k

        # Embed query
        embed_start_time = time.time()
        query_embedding = self.embedding_provider.embed_text(query)
        embed_time = time.time() - embed_start_time

        episodes = []
        facts = []
        summaries = []

        search_start_time = time.time()

        # Search episodes
        if search_episodes:
            episode_results = self._search_episodes(
                query_embedding, k, topic_filter, time_filter_since, time_filter_until
            )
            episodes = episode_results

        # Search facts
        if search_facts:
            fact_results = self._search_facts(query_embedding, k, topic_filter)
            facts = fact_results

        # Search summaries
        if search_summaries:
            summary_results = self._search_summaries(query_embedding, k, topic_filter)
            summaries = summary_results

        search_time = time.time() - search_start_time

        return SemanticResult(
            episodes=episodes,
            facts=facts,
            summaries=summaries,
            query_embedding_time=embed_time,
            search_time=search_time,
        )

    def _search_episodes(
        self,
        query_embedding: np.ndarray,
        k: int,
        topic_filter: Optional[str],
        time_since: Optional[datetime],
        time_until: Optional[datetime],
    ) -> list[Episode]:
        """Search episodes with optional topic/time filtering.

        Args:
            query_embedding: Embedded query vector.
            k: Maximum number of episodes to return.
            topic_filter: Optional topic constraint.
            time_since: Optional start time filter.
            time_until: Optional end time filter.

        Returns:
            A list of hydrated `Episode` objects.
        """

        # If filtering, get valid IDs first
        if topic_filter or time_since or time_until:
            db_episodes = self.database.get_episodes(
                topic=topic_filter,
                since=time_since,
                until=time_until,
                limit=k * 3,  # Get more to account for vector filtering
            )
            valid_ids = {ep.id for ep in db_episodes}

            if not valid_ids:
                return []

            results = self.vector_store.search_with_filter(
                "episodes", query_embedding, valid_ids, k=k, threshold=self.threshold
            )
        else:
            results = self.vector_store.search(
                "episodes", query_embedding, k=k, threshold=self.threshold
            )

        # Batch-fetch full episodes and filter out inactive ones
        record_ids = [record_id for record_id, _score in results]
        episodes_map = self.database.get_episodes_by_ids(record_ids)
        # Preserve vector-search ranking order
        episodes = [
            episodes_map[rid]
            for rid in record_ids
            if rid in episodes_map and episodes_map[rid].is_active
        ]

        return episodes

    def _search_facts(
        self,
        query_embedding: np.ndarray,
        k: int,
        topic_filter: Optional[str],
    ) -> list[Fact]:
        """Search facts with optional topic filtering.

        Args:
            query_embedding: Embedded query vector.
            k: Maximum number of facts to return.
            topic_filter: Optional topic constraint.

        Returns:
            A list of hydrated `Fact` objects.
        """

        if topic_filter:
            db_facts = self.database.get_facts(topic=topic_filter)
            valid_ids = {f.id for f in db_facts}

            if not valid_ids:
                return []

            results = self.vector_store.search_with_filter(
                "facts", query_embedding, valid_ids, k=k, threshold=self.threshold
            )
        else:
            results = self.vector_store.search(
                "facts", query_embedding, k=k, threshold=self.threshold
            )

        # Batch-fetch full facts and filter out inactive ones
        record_ids = [record_id for record_id, _score in results]
        facts_map = self.database.get_facts_by_ids(record_ids)
        facts = [
            facts_map[rid] for rid in record_ids if rid in facts_map and facts_map[rid].is_active
        ]

        return facts

    def _search_summaries(
        self,
        query_embedding: np.ndarray,
        k: int,
        topic_filter: Optional[str],
    ) -> list[Summary]:
        """Search summaries with optional topic filtering.

        Args:
            query_embedding: Embedded query vector.
            k: Maximum number of summaries to return.
            topic_filter: Optional topic constraint.

        Returns:
            A list of hydrated `Summary` objects.
        """

        if topic_filter:
            db_summaries = self.database.get_summaries(topic=topic_filter)
            valid_ids = {s.id for s in db_summaries}

            if not valid_ids:
                return []

            results = self.vector_store.search_with_filter(
                "summaries", query_embedding, valid_ids, k=k, threshold=self.threshold
            )
        else:
            results = self.vector_store.search(
                "summaries", query_embedding, k=k, threshold=self.threshold
            )

        # Batch-fetch full summaries and filter out inactive ones
        record_ids = [record_id for record_id, _score in results]
        summaries_map = self.database.get_summaries_by_ids(record_ids)
        summaries = [
            summaries_map[rid]
            for rid in record_ids
            if rid in summaries_map and summaries_map[rid].is_active
        ]

        return summaries

    def find_related_memories(
        self,
        episode: Episode,
        exclude_self: bool = True,
    ) -> SemanticResult:
        """Find memories semantically related to a given episode.

        Args:
            episode: Episode to use as the query seed.
            exclude_self: If True, remove the input episode from results.

        Returns:
            A `SemanticResult` containing items related to the episode.
        """
        # Use episode's embedding text as query
        query_text = episode.to_embedding_text()
        result = self.search(query_text)

        if exclude_self:
            result.episodes = [ep for ep in result.episodes if ep.id != episode.id]

        return result
