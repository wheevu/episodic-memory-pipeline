"""
Ingestion service for the episodic memory pipeline.

This module contains business logic for ingesting memories.
Returns plain dataclasses - no Rich/Typer imports.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from src.bootstrap import PipelineComponents
    from src.models import Episode


@dataclass
class IngestionResult:
    """Result of ingesting a piece of text."""

    success: bool
    episode: Optional["Episode"] = None
    reason: Optional[str] = None
    classification_confidence: Optional[float] = None


class IngestionService:
    """
    Service for ingesting memories into the pipeline.

    This service wraps the IngestionPipeline and provides a clean interface
    for use by CLI commands or other consumers.
    """

    def __init__(self, components: "PipelineComponents", worthiness_threshold: float = 0.6) -> None:
        """
        Initialize the ingestion service.

        Args:
            components: Pipeline components from bootstrap
            worthiness_threshold: Minimum score to store a memory

        Returns:
            None.
        """
        self.components = components
        self.worthiness_threshold = worthiness_threshold
        self._pipeline = None

    @property
    def pipeline(self) -> Any:
        """Lazily create the ingestion pipeline.

        Returns:
            An initialized ingestion pipeline instance.
        """
        if self._pipeline is None:
            self._pipeline = self.components.IngestionPipeline(
                self.components.database,
                self.components.vector_store,
                self.components.embedding_provider,
                self.components.llm,
                worthiness_threshold=self.worthiness_threshold,
            )
        return self._pipeline

    def ingest_text(self, text: str, source: str = "cli", force: bool = False) -> IngestionResult:
        """
        Ingest a piece of text into memory.

        Args:
            text: The text to ingest
            source: Source identifier
            force: Skip worthiness check

        Returns:
            IngestionResult with success status and details
        """
        result = self.pipeline.ingest(text, source=source, force=force)

        return IngestionResult(
            success=result.success,
            episode=result.episode if result.success else None,
            reason=result.reason if not result.success else None,
            classification_confidence=(
                result.classification.confidence if result.classification else None
            ),
        )

    def ingest_batch(
        self, texts: List[str], source: str = "batch", force: bool = False
    ) -> List[IngestionResult]:
        """
        Ingest multiple texts.

        Args:
            texts: List of texts to ingest
            source: Source identifier
            force: Skip worthiness check for all

        Returns:
            List of IngestionResult objects
        """
        results = []
        pipeline_results = self.pipeline.ingest_batch(
            texts,
            source=source,
            session_id=None,
            timestamp=None,
            force=force,
            persist_vectors=True,
        )

        for result in pipeline_results:
            results.append(
                IngestionResult(
                    success=result.success,
                    episode=result.episode if result.success else None,
                    reason=result.reason if not result.success else None,
                    classification_confidence=(
                        result.classification.confidence if result.classification else None
                    ),
                )
            )

        return results
