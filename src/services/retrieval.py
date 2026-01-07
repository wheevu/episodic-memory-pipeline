"""
Retrieval service for the episodic memory pipeline.

This module contains business logic for querying memories.
Returns plain dataclasses - no Rich/Typer imports.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.bootstrap import PipelineComponents
    from src.models import Episode, Fact


@dataclass
class QueryResult:
    """Result of a semantic query."""
    answer: Optional[str] = None
    confidence: float = 0.0
    episodes: List["Episode"] = field(default_factory=list)
    facts: List["Fact"] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


@dataclass
class NarrativeResult:
    """Result of a narrative recall."""
    answer: str = ""
    episodes: List["Episode"] = field(default_factory=list)
    topic: Optional[str] = None


@dataclass
class ConsolidationResult:
    """Result of consolidation operation."""
    topic: Optional[str]
    episodes_processed: int
    summaries_created: int
    facts_extracted: int
    duration_seconds: float


@dataclass
class SystemStats:
    """System statistics."""
    total_episodes: int
    unconsolidated_episodes: int
    total_facts: int
    total_summaries: int
    total_topics: int
    vector_stats: dict


class RetrievalService:
    """
    Service for retrieving memories from the pipeline.
    
    This service wraps the RetrievalEngine and ConsolidationPipeline
    and provides a clean interface for use by CLI commands.
    """
    
    def __init__(self, components: "PipelineComponents") -> None:
        """
        Initialize the retrieval service.
        
        Args:
            components: Pipeline components from bootstrap
        
        Returns:
            None.
        """
        self.components = components
        self._engine = None
        self._consolidation = None
    
    @property
    def engine(self) -> Any:
        """Lazily create the retrieval engine.

        Returns:
            An initialized retrieval engine instance.
        """
        if self._engine is None:
            self._engine = self.components.RetrievalEngine(
                self.components.database,
                self.components.vector_store,
                self.components.embedding_provider,
                self.components.llm
            )
        return self._engine
    
    @property
    def consolidation(self) -> Any:
        """Lazily create the consolidation pipeline.

        Returns:
            An initialized consolidation pipeline instance.
        """
        if self._consolidation is None:
            self._consolidation = self.components.ConsolidationPipeline(
                self.components.database,
                self.components.vector_store,
                self.components.embedding_provider,
                self.components.llm
            )
        return self._consolidation
    
    def query(self, query_text: str, synthesize: bool = True) -> QueryResult:
        """
        Query the memory system.
        
        Args:
            query_text: The query string
            synthesize: Whether to synthesize an answer
            
        Returns:
            QueryResult with answer and supporting evidence
        """
        result = self.engine.query(query_text, synthesize=synthesize)
        
        return QueryResult(
            answer=result.answer,
            confidence=result.confidence,
            episodes=result.episodes,
            facts=result.facts,
            gaps=result.gaps
        )
    
    def recall_narrative(
        self,
        topic_or_query: str,
        is_topic: bool = False
    ) -> NarrativeResult:
        """
        Recall a narrative for a topic.
        
        Args:
            topic_or_query: Topic name or query string
            is_topic: Whether to treat input as exact topic name
            
        Returns:
            NarrativeResult with narrative and timeline
        """
        result = self.engine.recall_narrative(topic_or_query, is_topic=is_topic)
        
        return NarrativeResult(
            answer=result.answer,
            episodes=result.episodes,
            topic=topic_or_query if is_topic else None
        )
    
    def consolidate(
        self,
        topic: Optional[str] = None,
        consolidate_all: bool = False
    ) -> List[ConsolidationResult]:
        """
        Run memory consolidation.
        
        Args:
            topic: Specific topic to consolidate
            consolidate_all: Consolidate all topics needing it
            
        Returns:
            List of ConsolidationResult objects
        """
        if topic:
            result = self.consolidation.consolidate_topic(topic)
            results = [result] if result else []
        elif consolidate_all:
            results = self.consolidation.consolidate_all()
        else:
            return []
        
        return [
            ConsolidationResult(
                topic=r.topic,
                episodes_processed=r.episodes_processed,
                summaries_created=r.summaries_created,
                facts_extracted=r.facts_extracted,
                duration_seconds=r.duration_seconds
            )
            for r in results
        ]
    
    def get_stats(self) -> SystemStats:
        """
        Get system statistics.
        
        Returns:
            SystemStats with database and vector store stats
        """
        db_stats = self.components.database.get_statistics()
        vec_stats = self.components.vector_store.get_statistics()
        
        return SystemStats(
            total_episodes=db_stats["total_episodes"],
            unconsolidated_episodes=db_stats["unconsolidated_episodes"],
            total_facts=db_stats["total_facts"],
            total_summaries=db_stats["total_summaries"],
            total_topics=db_stats["total_topics"],
            vector_stats=vec_stats
        )
    
    def get_topics(self, limit: int = 10) -> List[dict]:
        """
        Get list of topics.
        
        Args:
            limit: Maximum number of topics to return
            
        Returns:
            List of topic dictionaries
        """
        topics = self.components.database.get_topics()
        return topics[:limit] if topics else []

