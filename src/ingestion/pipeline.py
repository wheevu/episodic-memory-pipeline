"""
Ingestion pipeline - the main entry point for processing new memories.

Orchestrates: classification → extraction → embedding → storage
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from ..models import Episode
from ..storage import LanceStore
from .classifier import ClassificationResult, MemoryWorthinessClassifier
from .extractor import EpisodeExtractor, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Represents the outcome of ingesting a single input text.

    Attributes:
        success: Whether ingestion successfully stored an episode.
        episode: The stored episode, if any.
        classification: Classification stage output, if run.
        extraction: Extraction stage output, if run.
        reason: Human-readable reason for the outcome (stored/skipped).
    """

    success: bool
    episode: Optional[Episode] = None
    classification: Optional[ClassificationResult] = None
    extraction: Optional[ExtractionResult] = None
    reason: str = ""

    @classmethod
    def skipped(
        cls,
        reason: str,
        classification: Optional[ClassificationResult] = None,
    ) -> "IngestionResult":
        """Create a result representing a skipped ingestion.

        Args:
            reason: Explanation for why the input was not stored.
            classification: Classification output (if classification ran).

        Returns:
            An `IngestionResult` with `success=False`.
        """
        return cls(success=False, classification=classification, reason=reason)

    @classmethod
    def stored(
        cls,
        episode: Episode,
        classification: ClassificationResult,
        extraction: ExtractionResult,
    ) -> "IngestionResult":
        """Create a result representing a successfully stored episode.

        Args:
            episode: The stored episode.
            classification: Classification output used for the decision.
            extraction: Extraction output producing the episode structure.

        Returns:
            An `IngestionResult` with `success=True`.
        """
        return cls(
            success=True,
            episode=episode,
            classification=classification,
            extraction=extraction,
            reason="Stored successfully",
        )


class IngestionPipeline:
    """
    Main ingestion pipeline for processing raw input into stored memories.

    Pipeline stages:
    1. Classification: Is this worth remembering?
    2. Extraction: Extract structured memory
    3. Embedding: Generate vector embedding
    4. Storage: Persist to LanceDB store

    Design notes:
    - Each stage can short-circuit (e.g., not memory-worthy → skip rest)
    - Confidence scores propagate through the pipeline
    - Sessions allow grouping related inputs
    """

    def __init__(
        self,
        lance_store: LanceStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
        worthiness_threshold: float = 0.6,
    ) -> None:
        """Initialize the ingestion pipeline with all required dependencies.

        Args:
            lance_store: Unified metadata and vector store.
            embedding_provider: Provider used to embed episode text.
            llm: Provider used for classification and extraction.
            worthiness_threshold: Minimum classification confidence required to store.
        """
        self.lance_store = lance_store
        self.embedding_provider = embedding_provider
        self.classifier = MemoryWorthinessClassifier(llm, threshold=worthiness_threshold)
        self.extractor = EpisodeExtractor(llm)
        self.worthiness_threshold = worthiness_threshold

    def ingest(
        self,
        text: str,
        source: str = "chat",
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        context: Optional[str] = None,
        force: bool = False,
        persist_vectors: bool = True,
    ) -> IngestionResult:
        """Ingest a piece of text and (optionally) store it as an episode.

        Args:
            text: Raw input text to process.
            source: Source label for the input (e.g., "chat", "note", "import").
            session_id: Optional session identifier used to group related inputs.
            timestamp: When the input occurred; defaults to current UTC time.
            context: Optional extra context for classification.
            force: If True, bypass worthiness checks and store anyway.
            persist_vectors: Deprecated; retained for API compatibility.

        Returns:
            An `IngestionResult` describing whether storage occurred and why.
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Input validation
        MAX_TEXT_LENGTH = 50_000
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(
                "Rejected input: text length %d exceeds maximum %d", len(text), MAX_TEXT_LENGTH
            )
            return IngestionResult.skipped(
                f"Input too long ({len(text)} chars, max {MAX_TEXT_LENGTH})"
            )

        if not text or not text.strip():
            logger.warning("Rejected input: empty or whitespace-only text")
            return IngestionResult.skipped("Empty input")

        logger.debug("Ingesting text: source=%s, session=%s, force=%s", source, session_id, force)

        # Stage 1: Classification (unless forced)
        if not force:
            classification = self.classifier.classify(text, context=context, use_llm=True)

            logger.debug(
                "Classification: worthy=%s, confidence=%.2f, type=%s",
                classification.is_memory_worthy,
                classification.confidence,
                classification.memory_type,
            )

            if not classification.is_memory_worthy:
                logger.info("Skipped ingestion: %s", classification.reason)
                return IngestionResult.skipped(
                    f"Not memory-worthy: {classification.reason}", classification
                )

            if classification.confidence < self.worthiness_threshold:
                logger.info(
                    "Skipped ingestion: confidence %.2f below threshold %.2f",
                    classification.confidence,
                    self.worthiness_threshold,
                )
                return IngestionResult.skipped(
                    f"Below confidence threshold ({classification.confidence:.2f} < {self.worthiness_threshold})",
                    classification,
                )
        else:
            logger.debug("Bypassing classification (forced)")
            classification = ClassificationResult(
                is_memory_worthy=True,
                confidence=1.0,
                reason="Forced ingestion",
                memory_type="episodic",
            )

        # Stage 2: Extraction
        extraction = self.extractor.extract(
            text,
            memory_type_hint=classification.memory_type,
            timestamp=timestamp,
            source=source,
            session_id=session_id,
        )

        episode = extraction.episode

        # Update confidence from classification and extraction
        episode.confidence = min(classification.confidence, extraction.extraction_confidence)

        logger.debug(
            "Extracted episode: id=%s, type=%s, topics=%s",
            episode.id[:8],
            episode.memory_type,
            episode.topics,
        )

        # Stage 3: Embedding
        embedding_text = episode.to_embedding_text()
        embedding = self.embedding_provider.embed_text(embedding_text)

        # Stage 4: Storage (atomic metadata + vector write)
        try:
            self.lance_store.save_episode(episode, embedding)
            logger.debug("Saved episode to LanceDB store: %s", episode.id[:8])
        except Exception as exc:
            logger.error("LanceStore error for episode %s: %s", episode.id[:8], exc)
            return IngestionResult.skipped(f"Store error: {exc}", classification)

        logger.info(
            "Successfully ingested episode: id=%s, type=%s, importance=%.2f",
            episode.id[:8],
            episode.memory_type,
            episode.importance,
        )

        return IngestionResult.stored(episode, classification, extraction)

    def ingest_batch(
        self,
        texts: list[str],
        source: str = "chat",
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        force: bool = False,
        persist_vectors: bool = True,
    ) -> list[IngestionResult]:
        """Ingest multiple texts sequentially.

        Args:
            texts: Input texts to ingest.
            source: Source label applied to all inputs.
            session_id: Optional session identifier applied to all inputs.
            timestamp: Optional timestamp applied to all inputs.
            force: If True, bypass worthiness checks for all inputs.
            persist_vectors: Deprecated; retained for API compatibility.

        Returns:
            A list of `IngestionResult`, one per input text.
        """
        # Note: This remains sequential to preserve ordering and simplicity; embedding
        # could be batched later without changing the public API.
        results: list[IngestionResult] = []
        for text in texts:
            result = self.ingest(
                text,
                source=source,
                session_id=session_id,
                timestamp=timestamp,
                force=force,
                persist_vectors=False,
            )
            results.append(result)

        return results

    def get_statistics(self) -> dict:
        """Return unified store statistics.

        Returns:
            A dictionary of LanceStore stats.
        """
        return self.lance_store.get_statistics()
