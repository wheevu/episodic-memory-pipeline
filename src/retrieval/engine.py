"""
Retrieval engine - unified interface for memory queries.

Combines semantic and narrative retrieval with LLM-based answer synthesis.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from ..models import Episode, Fact, Summary
from ..prompts import PromptTemplates
from ..storage import LanceStore
from .narrative import NarrativeResult, NarrativeRetriever
from .semantic import SemanticResult, SemanticRetriever


@dataclass
class QueryResult:
    """Represents the end-to-end result of a user query.

    Attributes:
        answer: Final answer text (may be empty if synthesis disabled).
        confidence: Answer confidence score (heuristic or model-provided).
        episodes: Supporting episodes.
        facts: Supporting facts.
        summaries: Supporting summaries.
        query_type: Strategy used ("semantic" or "narrative").
        gaps: Missing information that would improve the answer.
    """

    answer: str
    confidence: float
    episodes: list[Episode]
    facts: list[Fact]
    summaries: list[Summary]
    query_type: str  # "semantic" or "narrative"
    gaps: list[str]  # Information that would help but is missing


class RetrievalEngine:
    """
    Unified retrieval engine that:
    1. Analyzes the query to determine best strategy
    2. Retrieves relevant memories
    3. Synthesizes an answer using LLM

    This is the main interface for querying the memory system.
    """

    def __init__(
        self,
        lance_store: LanceStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        """Initialize the retrieval engine.

        Args:
            lance_store: Unified metadata and vector store.
            embedding_provider: Provider used to embed user queries.
            llm: Provider used for query analysis and answer synthesis.
        """
        self.lance_store = lance_store
        self.llm = llm

        self.semantic = SemanticRetriever(lance_store, embedding_provider)
        self.narrative = NarrativeRetriever(lance_store, embedding_provider)

    def query(
        self,
        query: str,
        synthesize: bool = True,
    ) -> QueryResult:
        """Process a natural-language query end-to-end.

        This method (1) analyzes the query to choose a retrieval strategy,
        (2) retrieves relevant memories, then (3) optionally synthesizes an answer.

        Args:
            query: User's natural-language query.
            synthesize: If True, generate an LLM-based answer from retrieved context.

        Returns:
            A `QueryResult` containing retrieved context and (optionally) an answer.
        """
        # Analyze query to determine strategy
        query_analysis = self._analyze_query(query)

        query_type = query_analysis.get("query_type", "semantic")
        time_filter = query_analysis.get("time_filter", {})
        topic_filters = query_analysis.get("topic_filters", [])
        reformulated = query_analysis.get("reformulated_query", query)

        # Parse time filters
        since = None
        until = None
        if time_filter.get("since"):
            try:
                since = datetime.fromisoformat(time_filter["since"])
            except (ValueError, TypeError):
                pass
        if time_filter.get("until"):
            try:
                until = datetime.fromisoformat(time_filter["until"])
            except (ValueError, TypeError):
                pass

        # Retrieve based on query type
        if query_type == "narrative":
            result = self._narrative_retrieval(
                reformulated, topic_filters[0] if topic_filters else None, since, until
            )
        else:  # semantic or hybrid
            result = self._semantic_retrieval(
                reformulated, topic_filters[0] if topic_filters else None, since, until
            )

        # Synthesize answer if requested
        if synthesize:
            synthesis = self._synthesize_answer(query, result)
            answer = synthesis.get("answer", "I don't have enough information to answer this.")
            confidence = synthesis.get("confidence", 0.5)
            gaps = synthesis.get("gaps", [])
        else:
            answer = ""
            confidence = 0.0
            gaps = []

        return QueryResult(
            answer=answer,
            confidence=confidence,
            episodes=result.episodes,
            facts=result.facts if hasattr(result, "facts") else [],
            summaries=result.summaries if hasattr(result, "summaries") else [],
            query_type=query_type,
            gaps=gaps,
        )

    def _analyze_query(self, query: str) -> dict:
        """Analyze a query to determine retrieval strategy and filters.

        Args:
            query: User query text.

        Returns:
            A dictionary describing query type, time/topic filters, and a reformulation.
        """
        # Get known topics for context
        topics = self.lance_store.get_topics()
        topic_names = [t["name"] for t in topics]

        prompt = PromptTemplates.QUERY_ANALYSIS.format(
            query=query,
            known_topics=", ".join(topic_names[:20]) if topic_names else "none",
            recent_activity="recent memory activity",
        )

        try:
            return self.llm.complete_json(prompt)
        except (ValueError, Exception):
            # Default to semantic search
            return {
                "query_type": "semantic",
                "time_relevance": "all_time",
                "time_filter": {},
                "search_concepts": [query],
                "topic_filters": [],
                "reformulated_query": query,
            }

    def _semantic_retrieval(
        self,
        query: str,
        topic: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> SemanticResult:
        """Perform semantic retrieval via the `SemanticRetriever`.

        Args:
            query: Reformulated query text.
            topic: Optional topic filter.
            since: Optional start time filter.
            until: Optional end time filter.

        Returns:
            A `SemanticResult` containing retrieved items.
        """
        return self.semantic.search(
            query,
            topic_filter=topic,
            time_filter_since=since,
            time_filter_until=until,
        )

    def _narrative_retrieval(
        self,
        query: str,
        topic: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> NarrativeResult:
        """Perform narrative retrieval via the `NarrativeRetriever`.

        Args:
            query: Reformulated query text used when no explicit topic is available.
            topic: Optional explicit topic to recall.
            since: Optional start time filter.
            until: Optional end time filter.

        Returns:
            A `NarrativeResult` containing a chronologically ordered narrative.
        """
        if topic:
            return self.narrative.recall(topic, since=since, until=until)
        else:
            return self.narrative.recall_by_query(query, since=since, until=until)

    def _synthesize_answer(
        self,
        query: str,
        result: Any,
    ) -> dict:
        """Synthesize an answer from retrieved memories using the LLM.

        Args:
            query: Original user query (not reformulated).
            result: Retrieval result object (semantic or narrative).

        Returns:
            A dictionary containing the synthesized answer, confidence, and gaps.
        """
        # Format memories for prompt
        episodes_text = PromptTemplates.format_episodes_for_prompt(
            result.episodes if result.episodes else []
        )

        facts_text = PromptTemplates.format_facts_for_prompt(
            result.facts if hasattr(result, "facts") and result.facts else []
        )

        summaries_text = PromptTemplates.format_summaries_for_prompt(
            result.summaries if hasattr(result, "summaries") and result.summaries else []
        )

        prompt = PromptTemplates.ANSWER_SYNTHESIS.format(
            query=query, summaries=summaries_text, facts=facts_text, episodes=episodes_text
        )

        try:
            return self.llm.complete_json(prompt)
        except (ValueError, Exception) as e:
            return {
                "answer": f"Retrieved {len(result.episodes)} episodes but couldn't synthesize answer: {e}",
                "confidence": 0.3,
                "key_sources": [],
                "gaps": ["synthesis failed"],
            }

    def recall_narrative(
        self,
        topic_or_query: str,
        is_topic: bool = False,
    ) -> QueryResult:
        """Recall a narrative (story/journey) about a topic or inferred topic.

        This is optimized for "Tell me about..." style requests and returns a
        narrative synthesis as the answer.

        Args:
            topic_or_query: Topic name (if `is_topic=True`) or free-form query text.
            is_topic: If True, treat `topic_or_query` as an explicit topic.

        Returns:
            A `QueryResult` with narrative answer and supporting context.
        """
        if is_topic:
            result = self.narrative.recall(topic_or_query)
        else:
            result = self.narrative.recall_by_query(topic_or_query)

        # Generate narrative synthesis
        narrative = self._synthesize_narrative(topic_or_query, result)

        return QueryResult(
            answer=narrative.get("narrative", "No narrative available."),
            confidence=0.8 if result.episodes else 0.2,
            episodes=result.episodes,
            facts=result.facts,
            summaries=result.summaries,
            query_type="narrative",
            gaps=[],
        )

    def _synthesize_narrative(
        self,
        topic: str,
        result: NarrativeResult,
    ) -> dict:
        """Generate a narrative synthesis from time-ordered memories.

        Args:
            topic: Topic label used in the narrative prompt.
            result: Narrative retrieval output to synthesize from.

        Returns:
            A dictionary containing narrative text and optional structured fields.
        """
        episodes_text = PromptTemplates.format_episodes_for_prompt(result.episodes)
        facts_text = PromptTemplates.format_facts_for_prompt(result.facts)
        summaries_text = PromptTemplates.format_summaries_for_prompt(result.summaries)

        prompt = PromptTemplates.NARRATIVE_SYNTHESIS.format(
            topic=topic,
            query=f"Tell me about {topic}",
            episodes=episodes_text,
            facts=facts_text,
            summaries=summaries_text,
        )

        try:
            return self.llm.complete_json(prompt)
        except (ValueError, Exception):
            return {
                "narrative": f"Found {len(result.episodes)} memories about {topic}.",
                "timeline": [],
                "key_moments": [],
                "current_status": "Unknown",
            }

    def quick_lookup(self, query: str) -> list[Fact]:
        """Perform a quick fact-only lookup without answer synthesis.

        Args:
            query: Fact-oriented query text (e.g., "What is my favorite food?").

        Returns:
            A list of facts returned by semantic search.
        """
        result = self.semantic.search(
            query, search_episodes=False, search_summaries=False, search_facts=True, top_k=5
        )
        return result.facts

    def get_context(self, topic: str, max_items: int = 5) -> dict:
        """Return a compact context bundle for a topic.

        Args:
            topic: Topic to fetch context for.
            max_items: Maximum number of episodes/facts to include.

        Returns:
            A dictionary containing recent episodes, facts, and the latest summary.
        """
        episodes = self.lance_store.get_episodes(topic=topic, limit=max_items)
        facts = self.lance_store.get_facts(topic=topic, limit=max_items)
        summary = self.lance_store.get_latest_summary(topic)

        return {
            "topic": topic,
            "recent_episodes": episodes,
            "facts": facts,
            "summary": summary,
            "episode_count": len(episodes),
            "fact_count": len(facts),
        }
