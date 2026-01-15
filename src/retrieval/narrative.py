"""
Narrative retrieval - story/timeline based memory recall.

For queries like "What have I said about learning Korean?" or
"Tell me about my journey with X"
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..embeddings import EmbeddingProvider
from ..models import Episode, Fact, Summary
from ..storage import Database, VectorStore


@dataclass
class NarrativeResult:
    """Represents results of a narrative/timeline retrieval query.

    Attributes:
        episodes: Episodes ordered chronologically (oldest first).
        facts: Facts associated with the inferred/selected topic.
        summaries: Summaries associated with the inferred/selected topic.
        topic: Topic used (explicit or inferred), if any.
        time_span: Inclusive (start, end) bounds of the episode timeline, if any.
    """

    episodes: list[Episode]  # Time-ordered
    facts: list[Fact]
    summaries: list[Summary]
    topic: Optional[str]
    time_span: Optional[tuple[datetime, datetime]]


class NarrativeRetriever:
    """
    Narrative retrieval for timeline/story-based queries.

    Prioritizes:
    - Chronological ordering
    - Topic coherence
    - Key events from summaries
    - Complete story arcs
    """

    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        max_episodes: int = 50,
    ) -> None:
        """Initialize the narrative retriever.

        Args:
            database: Structured storage used to fetch episodes/facts/summaries.
            vector_store: Vector index used for semantic matching.
            embedding_provider: Provider used to embed free-form queries.
            max_episodes: Maximum episodes to include in a narrative timeline.
        """
        self.database = database
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.max_episodes = max_episodes

    def recall(
        self,
        topic: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        include_summaries: bool = True,
        include_facts: bool = True,
    ) -> NarrativeResult:
        """Recall a topic narrative within an optional time window.

        Args:
            topic: Topic to recall.
            since: Optional start time for episode selection.
            until: Optional end time for episode selection.
            include_summaries: If True, include summaries for the topic.
            include_facts: If True, include facts for the topic.

        Returns:
            A `NarrativeResult` with chronologically ordered memories.
        """
        # Get episodes for topic, ordered by time
        episodes = self.database.get_episodes(
            topic=topic, since=since, until=until, limit=self.max_episodes
        )

        # Sort chronologically (oldest first for narrative)
        episodes = sorted(episodes, key=lambda e: e.occurred_at)

        # Get time span
        time_span = None
        if episodes:
            time_span = (episodes[0].occurred_at, episodes[-1].occurred_at)

        # Get summaries
        summaries = []
        if include_summaries:
            summaries = self.database.get_summaries(topic=topic, since=since)
            summaries = sorted(summaries, key=lambda s: s.time_start)

        # Get facts
        facts = []
        if include_facts:
            facts = self.database.get_facts(topic=topic)

        return NarrativeResult(
            episodes=episodes, facts=facts, summaries=summaries, topic=topic, time_span=time_span
        )

    def recall_by_query(
        self,
        query: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> NarrativeResult:
        """Recall a narrative matching a query, inferring the topic from results.

        Uses semantic search to find relevant episodes, then expands to a full
        topic narrative when a clear primary topic can be inferred.

        Args:
            query: Natural language query used to seed semantic retrieval.
            since: Optional start time filter for narrative expansion.
            until: Optional end time filter for narrative expansion.

        Returns:
            A `NarrativeResult` for the inferred topic, or an empty result.
        """
        # Embed query
        query_embedding = self.embedding_provider.embed_text(query)

        # Search episodes
        results = self.vector_store.search("episodes", query_embedding, k=20, threshold=0.6)

        if not results:
            return NarrativeResult(episodes=[], facts=[], summaries=[], topic=None, time_span=None)

        # Fetch episodes and infer topic
        episode_ids = [rid for rid, _ in results]
        episodes = [self.database.get_episode(eid) for eid in episode_ids]
        episodes = [ep for ep in episodes if ep is not None]

        if not episodes:
            return NarrativeResult(episodes=[], facts=[], summaries=[], topic=None, time_span=None)

        # Infer primary topic from episodes
        topic_counts: dict[str, int] = {}
        for ep in episodes:
            for t in ep.topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1

        primary_topic = max(topic_counts, key=topic_counts.get) if topic_counts else None

        # If we have a clear topic, expand to full narrative
        if primary_topic and topic_counts[primary_topic] >= 2:
            # Get more episodes for this topic
            full_episodes = self.database.get_episodes(
                topic=primary_topic, since=since, until=until, limit=self.max_episodes
            )
            episodes = sorted(full_episodes, key=lambda e: e.occurred_at)
        else:
            # Just use semantic results, sorted by time
            episodes = sorted(episodes, key=lambda e: e.occurred_at)

        # Get related content
        summaries = []
        facts = []
        if primary_topic:
            summaries = self.database.get_summaries(topic=primary_topic, since=since)
            summaries = sorted(summaries, key=lambda s: s.time_start)
            facts = self.database.get_facts(topic=primary_topic)

        time_span = None
        if episodes:
            time_span = (episodes[0].occurred_at, episodes[-1].occurred_at)

        return NarrativeResult(
            episodes=episodes,
            facts=facts,
            summaries=summaries,
            topic=primary_topic,
            time_span=time_span,
        )

    def get_recent_journey(
        self,
        topic: str,
        days: int = 30,
    ) -> NarrativeResult:
        """Return a recent narrative slice for a topic.

        Args:
            topic: Topic to recall.
            days: Lookback window size in days.

        Returns:
            A `NarrativeResult` constrained to the lookback period.
        """
        since = datetime.utcnow() - timedelta(days=days)
        return self.recall(topic, since=since)

    def get_key_moments(
        self,
        topic: str,
        limit: int = 10,
    ) -> list[Episode]:
        """Return the most important episodes for a topic.

        This uses each episode's intrinsic `importance` score and applies a small
        boost for episodes referenced in summaries (treated as key events).

        Args:
            topic: Topic to select key episodes from.
            limit: Maximum number of episodes to return.

        Returns:
            A list of episodes ordered by descending importance.
        """
        # Get summaries to find key events
        summaries = self.database.get_summaries(topic=topic)

        key_episode_ids = set()
        for summary in summaries:
            key_episode_ids.update(summary.source_episode_ids)

        # Get episodes and sort by importance
        episodes = self.database.get_episodes(topic=topic, limit=100)

        # Boost episodes that are in key_events
        def score(ep: Episode) -> float:
            """Compute a ranking score for a key-moment candidate episode."""
            base = ep.importance
            if ep.id in key_episode_ids:
                base += 0.2
            return base

        episodes = sorted(episodes, key=score, reverse=True)
        return episodes[:limit]

    def build_timeline(
        self,
        topic: str,
        granularity: str = "day",
    ) -> list[dict]:
        """Build a grouped timeline view of a topic's episodes.

        Args:
            topic: Topic to build a timeline for.
            granularity: Grouping unit: "day", "week", or "month".

        Returns:
            A list of timeline buckets, each containing a period label and events.
        """
        episodes = self.database.get_episodes(topic=topic, limit=200)
        episodes = sorted(episodes, key=lambda e: e.occurred_at)

        if not episodes:
            return []

        timeline = []
        current_period = None
        current_events = []

        for ep in episodes:
            # Determine period
            if granularity == "day":
                period = ep.occurred_at.date()
            elif granularity == "week":
                period = ep.occurred_at.isocalendar()[:2]  # (year, week)
            else:  # month
                period = (ep.occurred_at.year, ep.occurred_at.month)

            if period != current_period:
                if current_events:
                    timeline.append(
                        {
                            "period": str(current_period),
                            "events": current_events,
                            "count": len(current_events),
                        }
                    )
                current_period = period
                current_events = []

            current_events.append({"id": ep.id, "content": ep.content, "importance": ep.importance})

        # Don't forget the last period
        if current_events:
            timeline.append(
                {
                    "period": str(current_period),
                    "events": current_events,
                    "count": len(current_events),
                }
            )

        return timeline
