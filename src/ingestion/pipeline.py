"""
Ingestion pipeline - the main entry point for processing new memories.

Orchestrates: classification → extraction → embedding → storage
"""
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..models import Episode
from ..storage import Database, VectorStore
from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from .classifier import MemoryWorthinessClassifier, ClassificationResult
from .extractor import EpisodeExtractor, ExtractionResult


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
        return cls(
            success=False,
            classification=classification,
            reason=reason
        )
    
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
            reason="Stored successfully"
        )


class IngestionPipeline:
    """
    Main ingestion pipeline for processing raw input into stored memories.
    
    Pipeline stages:
    1. Classification: Is this worth remembering?
    2. Extraction: Extract structured memory
    3. Embedding: Generate vector embedding
    4. Storage: Persist to database and vector store
    
    Design notes:
    - Each stage can short-circuit (e.g., not memory-worthy → skip rest)
    - Confidence scores propagate through the pipeline
    - Sessions allow grouping related inputs
    """
    
    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
        worthiness_threshold: float = 0.6,
    ) -> None:
        """Initialize the ingestion pipeline with all required dependencies.

        Args:
            database: Structured storage for episodes/facts/summaries.
            vector_store: Vector index for semantic retrieval.
            embedding_provider: Provider used to embed episode text.
            llm: Provider used for classification and extraction.
            worthiness_threshold: Minimum classification confidence required to store.
        """
        self.database = database
        self.vector_store = vector_store
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
            persist_vectors: If True, persist vector indexes to disk.

        Returns:
            An `IngestionResult` describing whether storage occurred and why.
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Stage 1: Classification (unless forced)
        if not force:
            classification = self.classifier.classify(
                text,
                context=context,
                use_llm=True
            )
            
            if not classification.is_memory_worthy:
                return IngestionResult.skipped(
                    f"Not memory-worthy: {classification.reason}",
                    classification
                )
            
            if classification.confidence < self.worthiness_threshold:
                return IngestionResult.skipped(
                    f"Below confidence threshold ({classification.confidence:.2f} < {self.worthiness_threshold})",
                    classification
                )
        else:
            classification = ClassificationResult(
                is_memory_worthy=True,
                confidence=1.0,
                reason="Forced ingestion",
                memory_type="episodic"
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
        episode.confidence = min(
            classification.confidence,
            extraction.extraction_confidence
        )
        
        # Stage 3: Embedding
        embedding_text = episode.to_embedding_text()
        embedding = self.embedding_provider.embed_text(embedding_text)
        
        # Stage 4: Storage
        # Save episode first (embedding_id set after vector insert)
        self.database.save_episode(episode)
        
        try:
            embedding_id = self.vector_store.add(
                "episodes",
                episode.id,
                embedding
            )
        except Exception as exc:
            self.database.set_episode_active(episode.id, False)
            return IngestionResult.skipped(
                f"Vector store error: {exc}",
                classification
            )
        
        # Update episode with embedding ID
        episode.embedding_id = embedding_id
        try:
            self.database.update_embedding_id("episodes", episode.id, embedding_id)
        except Exception as exc:
            self.database.set_episode_active(episode.id, False)
            return IngestionResult.skipped(
                f"Embedding update error: {exc}",
                classification
            )
        
        # Persist vector store
        if persist_vectors:
            self.vector_store.save()
        
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
            persist_vectors: If True, persist vector indexes once at the end.

        Returns:
            A list of `IngestionResult`, one per input text.
        """
        # Note: This remains sequential to preserve ordering and simplicity; embedding
        # could be batched later without changing the public API.
        results: list[IngestionResult] = []
        stored_any = False
        for text in texts:
            result = self.ingest(
                text,
                source=source,
                session_id=session_id,
                timestamp=timestamp,
                force=force,
                persist_vectors=False,
            )
            if result.success:
                stored_any = True
            results.append(result)

        if persist_vectors and stored_any:
            self.vector_store.save()

        return results
    
    def get_statistics(self) -> dict:
        """Return basic database and vector-store statistics.

        Returns:
            A dictionary with `database` and `vector_store` stats payloads.
        """
        db_stats = self.database.get_statistics()
        vec_stats = self.vector_store.get_statistics()
        return {
            "database": db_stats,
            "vector_store": vec_stats,
        }

