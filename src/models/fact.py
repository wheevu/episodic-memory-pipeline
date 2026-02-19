"""
Fact model - represents semantic memory (stable knowledge).

A fact is a distilled piece of knowledge extracted from one or more episodes.
Facts are more durable than episodes and represent what we "know" vs what "happened".
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class FactCategory(str, Enum):
    """Categories of factual knowledge."""

    PERSONAL = "personal"  # About the user
    PREFERENCE = "preference"  # User preferences
    RELATIONSHIP = "relationship"  # People/entities user knows
    KNOWLEDGE = "knowledge"  # Things user knows
    CONTEXT = "context"  # Situational context
    GOAL = "goal"  # Long-term goals


class Fact(BaseModel):
    """
    A semantic memory - a stable piece of knowledge.

    Design notes:
    - Facts are extracted from episodes via consolidation
    - They have temporal validity (valid_from, valid_until)
    - Supersession allows tracking how facts evolve over time
    - Confidence degrades if not reinforced by new episodes
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Content
    content: str  # The factual statement
    category: FactCategory = FactCategory.PERSONAL

    # Metadata
    topic: str  # Primary topic
    entities: list[str] = Field(default_factory=list)

    # Confidence and validity
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None  # NULL = currently valid

    # State
    is_active: bool = True
    superseded_by: Optional[str] = None  # ID of newer fact

    # Provenance (populated on retrieval)
    source_episode_ids: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True

    @property
    def is_current(self) -> bool:
        """Return True if the fact is currently valid.

        Returns:
            True if within validity window, active, and not superseded.
        """
        now = datetime.now(timezone.utc)
        if self.valid_until and self.valid_until < now:
            return False
        if self.valid_from and self.valid_from > now:
            return False
        return self.is_active and self.superseded_by is None

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding.

        Returns:
            A compact string representation used for embedding generation.
        """
        parts = [self.content]
        parts.append(f"Category: {self.category}")
        parts.append(f"Topic: {self.topic}")
        return " | ".join(parts)
