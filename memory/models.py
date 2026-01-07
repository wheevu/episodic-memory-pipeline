"""
Data models for the episodic memory system.

Three-tier memory architecture:
- Episode: Raw, timestamped events (episodic memory)
- Fact: Stable, extracted knowledge (semantic memory)  
- Summary: Consolidated narratives (narrative memory)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List
import json
import uuid


class MemoryType(str, Enum):
    """Classification of memory content."""
    EPISODIC = "episodic"      # Events, conversations, experiences
    FACT = "fact"              # Learned information, knowledge
    GOAL = "goal"              # Intentions, plans, aspirations
    PREFERENCE = "preference"  # Likes, dislikes, personal choices
    REFLECTION = "reflection"  # Meta-cognition, insights about self


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""
    ACTIVE = "active"          # Current, relevant memory
    CONSOLIDATED = "consolidated"  # Merged into a summary
    SUPERSEDED = "superseded"  # Replaced by newer information
    ARCHIVED = "archived"      # Old but preserved for history


@dataclass
class Episode:
    """
    An episodic memory - a timestamped record of an event or interaction.
    
    Episodes are the raw material of memory. They preserve:
    - The original content/context
    - When it happened
    - What type of memory it represents
    - Structured extraction from the content
    """
    id: str
    content: str                    # Original text/interaction
    memory_type: MemoryType
    created_at: datetime
    
    # Structured extraction
    extracted_info: dict = field(default_factory=dict)  # LLM-extracted structure
    topics: List[str] = field(default_factory=list)     # Topic tags
    entities: List[str] = field(default_factory=list)   # Named entities mentioned
    
    # Metadata
    confidence: float = 1.0         # Extraction confidence (0-1)
    source: str = "user_input"      # Where this came from
    status: MemoryStatus = MemoryStatus.ACTIVE
    
    # Embedding reference (stored separately in FAISS)
    embedding_id: Optional[int] = None
    
    # Provenance
    session_id: Optional[str] = None  # Group related episodes
    
    @classmethod
    def create(
        cls,
        content: str,
        memory_type: MemoryType,
        extracted_info: dict = None,
        topics: List[str] = None,
        entities: List[str] = None,
        confidence: float = 1.0,
        source: str = "user_input",
        session_id: str = None,
    ) -> "Episode":
        """Factory method for creating new episodes.

        Args:
            content: Episode content (raw user text or cleaned memory content).
            memory_type: Classification of the memory.
            extracted_info: Optional structured extraction payload (LLM output).
            topics: Optional topic tags.
            entities: Optional named entities.
            confidence: Extraction confidence score in [0.0, 1.0].
            source: Source label for provenance.
            session_id: Optional session identifier for grouping.

        Returns:
            Newly created `Episode` instance.
        """
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            created_at=datetime.utcnow(),
            extracted_info=extracted_info or {},
            topics=topics or [],
            entities=entities or [],
            confidence=confidence,
            source=source,
            session_id=session_id,
        )
    
    def to_embedding_text(self) -> str:
        """Generate text for embedding - combines content with structure.

        Returns:
            A string suitable for embedding generation.
        """
        parts = [self.content]
        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")
        if self.extracted_info.get("summary"):
            parts.append(f"Summary: {self.extracted_info['summary']}")
        return "\n".join(parts)


@dataclass
class Fact:
    """
    A semantic memory - a stable piece of extracted knowledge.
    
    Facts are distilled from episodes. They represent:
    - Persistent truths about the user/world
    - Information that should be recalled without temporal context
    - Knowledge that can be updated or contradicted
    """
    id: str
    content: str                    # The fact statement
    topic: str                      # Primary topic category
    created_at: datetime
    updated_at: datetime
    
    # Provenance - which episodes support this fact
    source_episode_ids: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 1.0         # How confident we are in this fact
    fact_type: str = "general"      # Subcategory (e.g., "preference", "skill", "relationship")
    status: MemoryStatus = MemoryStatus.ACTIVE
    
    # Embedding reference
    embedding_id: Optional[int] = None
    
    # Conflict tracking
    superseded_by: Optional[str] = None  # ID of fact that replaced this one
    
    @classmethod
    def create(
        cls,
        content: str,
        topic: str,
        source_episode_ids: List[str],
        fact_type: str = "general",
        confidence: float = 1.0,
    ) -> "Fact":
        """Factory method for creating new facts.

        Args:
            content: Fact statement.
            topic: Primary topic category.
            source_episode_ids: Episode IDs supporting this fact.
            fact_type: Optional fact subcategory (e.g., preference, relationship).
            confidence: Confidence score in [0.0, 1.0].

        Returns:
            Newly created `Fact` instance.
        """
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            topic=topic,
            created_at=now,
            updated_at=now,
            source_episode_ids=source_episode_ids,
            fact_type=fact_type,
            confidence=confidence,
        )
    
    def add_source(self, episode_id: str) -> None:
        """Add a supporting episode to this fact.

        Args:
            episode_id: Episode ID to add as supporting evidence.

        Returns:
            None.
        """
        if episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
            self.updated_at = datetime.utcnow()


@dataclass
class Summary:
    """
    A narrative memory - a consolidated summary of related episodes.
    
    Summaries provide:
    - Compressed representation of many episodes
    - Temporal narrative flow
    - Topic-level overview without losing the ability to drill down
    """
    id: str
    content: str                    # The summary text
    topic: str                      # What topic this summarizes
    created_at: datetime
    updated_at: datetime
    
    # Time range covered
    period_start: datetime
    period_end: datetime
    
    # Provenance - episodes that went into this summary
    source_episode_ids: List[str] = field(default_factory=list)
    
    # Extracted facts from this consolidation
    extracted_fact_ids: List[str] = field(default_factory=list)
    
    # Metadata
    episode_count: int = 0          # How many episodes were consolidated
    status: MemoryStatus = MemoryStatus.ACTIVE
    
    # Embedding reference
    embedding_id: Optional[int] = None
    
    # Hierarchy - summaries can be consolidated into higher-level summaries
    parent_summary_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        content: str,
        topic: str,
        source_episode_ids: List[str],
        period_start: datetime,
        period_end: datetime,
    ) -> "Summary":
        """Factory method for creating new summaries.

        Args:
            content: Summary text.
            topic: Topic this summary covers.
            source_episode_ids: Episode IDs consolidated into this summary.
            period_start: Start time of the covered period.
            period_end: End time of the covered period.

        Returns:
            Newly created `Summary` instance.
        """
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            topic=topic,
            created_at=now,
            updated_at=now,
            period_start=period_start,
            period_end=period_end,
            source_episode_ids=source_episode_ids,
            episode_count=len(source_episode_ids),
        )


@dataclass
class RetrievalResult:
    """Result from a memory retrieval query."""
    query: str
    mode: str  # "semantic" or "narrative"
    
    # The synthesized answer
    answer: str
    
    # Supporting evidence
    episodes: List[Episode] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    summaries: List[Summary] = field(default_factory=list)
    
    # Scores
    relevance_scores: dict = field(default_factory=dict)  # id -> score
    
    # Metadata
    total_results: int = 0
    retrieval_time_ms: float = 0


@dataclass
class ExtractionResult:
    """Result from LLM memory extraction."""
    should_store: bool              # Whether this is memory-worthy
    memory_type: MemoryType
    confidence: float
    
    # Extracted structure
    summary: str                    # Brief summary of the content
    topics: List[str]
    entities: List[str]
    
    # For facts/goals/preferences
    key_points: List[str] = field(default_factory=list)
    
    # Reasoning
    extraction_reasoning: str = ""  # Why we extracted what we did
    rejection_reason: str = ""      # If not storing, why not

