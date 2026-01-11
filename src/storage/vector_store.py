"""
FAISS vector store for semantic similarity search.

Manages vector embeddings for episodes, facts, and summaries.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu package required. Install with: pip install faiss-cpu")


@dataclass
class SearchResult:
    """Represents a single vector similarity match returned by FAISS.

    Attributes:
        id: FAISS internal index ID for the match.
        score: Similarity score (higher is more similar for inner-product indices).
        distance: Distance metric (lower is more similar for L2 indices).
    """
    id: int              # FAISS index ID
    score: float         # Similarity score (higher = more similar)
    distance: float      # L2 distance (lower = more similar)


class VectorStore:
    """
    FAISS-based vector store for similarity search.
    
    Uses a flat L2 index for simplicity. For larger datasets,
    consider IVF or HNSW indices.
    
    Design notes:
    - Maintains separate indices for episodes, facts, summaries
    - Index IDs map to database record IDs via metadata
    - Supports incremental additions and persistence
    """
    
    def __init__(self, base_path: Path, dimension: int = 1536, auto_save: bool = False) -> None:
        """Initialize a FAISS-backed vector store with per-entity indices.

        Args:
            base_path: Base path used to derive index filenames on disk.
            dimension: Embedding dimensionality expected by all indices.
            auto_save: If True, automatically persist after each modification.
        """
        self.base_path = base_path
        self.dimension = dimension
        self.auto_save = auto_save
        
        # Ensure directory exists
        try:
            self.base_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create vector store directory: %s", e)
            raise
        
        # Initialize or load indices
        self._indices: dict[str, faiss.Index] = {}
        self._id_maps: dict[str, list[str]] = {}  # Maps FAISS IDs to record IDs
        self._dirty: set[str] = set()  # Track which indices have unsaved changes
        
        self._load_or_create_indices()
    
    def _index_path(self, name: str) -> Path:
        """Return the on-disk path for the FAISS index with the given name.

        Args:
            name: Index name (e.g., "episodes", "facts", "summaries").

        Returns:
            Full filesystem path for the `.faiss` index file.
        """
        return self.base_path.parent / f"{self.base_path.stem}_{name}.faiss"
    
    def _id_map_path(self, name: str) -> Path:
        """Return the on-disk path for the record-ID mapping file.

        Args:
            name: Index name (e.g., "episodes", "facts", "summaries").

        Returns:
            Full filesystem path for the `.npy` ID map file.
        """
        return self.base_path.parent / f"{self.base_path.stem}_{name}_ids.npy"
    
    def _load_or_create_indices(self) -> None:
        """Load existing indices from disk or create fresh empty indices."""
        for name in ["episodes", "facts", "summaries"]:
            index_path = self._index_path(name)
            id_map_path = self._id_map_path(name)
            
            if index_path.exists():
                self._indices[name] = faiss.read_index(str(index_path))
                if id_map_path.exists():
                    self._id_maps[name] = list(np.load(str(id_map_path), allow_pickle=True))
                else:
                    self._id_maps[name] = []

                if self._indices[name].d != self.dimension:
                    logger.error(
                        "Index '%s' dimension mismatch: expected %d but found %d. "
                        "Resetting index (all vectors will be lost). "
                        "To avoid this, ensure EMBEDDING_DIMENSION matches your model.",
                        name,
                        self.dimension,
                        self._indices[name].d,
                    )
                    self._indices[name] = faiss.IndexFlatIP(self.dimension)
                    self._id_maps[name] = []
                    self._dirty.add(name)  # Mark as dirty to force persistence

                if len(self._id_maps[name]) != self._indices[name].ntotal:
                    logger.error(
                        "Index '%s' id map mismatch: %d IDs but %d vectors. "
                        "This indicates corruption. Resetting index (all vectors will be lost).",
                        name,
                        len(self._id_maps[name]),
                        self._indices[name].ntotal,
                    )
                    self._indices[name] = faiss.IndexFlatIP(self.dimension)
                    self._id_maps[name] = []
                    self._dirty.add(name)  # Mark as dirty to force persistence
            else:
                # Create flat L2 index (exact search)
                # Use inner-product on unit-normalized vectors to compute cosine similarity.
                self._indices[name] = faiss.IndexFlatIP(self.dimension)
                self._id_maps[name] = []
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product corresponds to cosine similarity.

        Args:
            vectors: Array of vectors shaped (n, d).

        Returns:
            Normalized array with the same shape as `vectors`.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add(
        self,
        index_name: str,
        record_id: str,
        embedding: np.ndarray
    ) -> int:
        """Add a single embedding vector to the named index.

        Args:
            index_name: Index name ("episodes", "facts", or "summaries").
            record_id: Database record identifier associated with this vector.
            embedding: Embedding vector; will be cast to `np.float32` and normalized.

        Returns:
            The FAISS internal ID assigned to the newly added vector.

        Raises:
            ValueError: If `index_name` is not a known index.
        """
        if index_name not in self._indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        # Ensure correct shape and normalize
        embedding = embedding.reshape(1, -1).astype(np.float32)
        embedding = self._normalize(embedding)
        
        # Add to index
        faiss_id = self._indices[index_name].ntotal
        self._indices[index_name].add(embedding)
        self._id_maps[index_name].append(record_id)
        self._dirty.add(index_name)
        
        if self.auto_save:
            self.save(index_name)
        
        return faiss_id
    
    def add_batch(
        self,
        index_name: str,
        record_ids: list[str],
        embeddings: np.ndarray
    ) -> list[int]:
        """Add multiple embedding vectors to the named index.

        Args:
            index_name: Index name ("episodes", "facts", or "summaries").
            record_ids: Database record identifiers associated with each vector.
            embeddings: Embedding matrix shaped (n, d); will be cast and normalized.

        Returns:
            A list of FAISS IDs corresponding to the added vectors.

        Raises:
            ValueError: If `index_name` is not a known index.
            ValueError: If `record_ids` length does not match number of embeddings.
        """
        if index_name not in self._indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        if len(record_ids) != embeddings.shape[0]:
            raise ValueError("record_ids and embeddings must have same length")
        
        # Normalize
        embeddings = embeddings.astype(np.float32)
        embeddings = self._normalize(embeddings)
        
        # Record starting ID
        start_id = self._indices[index_name].ntotal
        
        # Add to index
        self._indices[index_name].add(embeddings)
        self._id_maps[index_name].extend(record_ids)
        self._dirty.add(index_name)
        
        if self.auto_save:
            self.save(index_name)
        
        return list(range(start_id, start_id + len(record_ids)))
    
    def search(
        self,
        index_name: str,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> list[tuple[str, float]]:
        """Search for vectors similar to `query_embedding`.

        Args:
            index_name: Index name ("episodes", "facts", or "summaries").
            query_embedding: Query embedding vector; will be cast and normalized.
            k: Maximum number of results to return.
            threshold: Optional minimum similarity threshold (cosine similarity).

        Returns:
            A list of `(record_id, similarity_score)` tuples.

        Raises:
            ValueError: If `index_name` is not a known index.
        """
        if index_name not in self._indices:
            raise ValueError(f"Unknown index: {index_name}")
        
        index = self._indices[index_name]
        if index.ntotal == 0:
            return []
        
        # Prepare query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = self._normalize(query)
        
        # Search
        k = min(k, index.ntotal)
        scores, indices = index.search(query, k)
        
        # Map to record IDs and filter by threshold
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            if threshold is not None and score < threshold:
                continue
            
            record_id = self._id_maps[index_name][idx]
            results.append((record_id, float(score)))
        
        return results
    
    def search_with_filter(
        self,
        index_name: str,
        query_embedding: np.ndarray,
        valid_ids: set[str],
        k: int = 10,
        threshold: Optional[float] = None
    ) -> list[tuple[str, float]]:
        """Search while restricting matches to a given allowlist of record IDs.

        This uses a post-filter strategy (search broadly, then filter), which avoids
        rebuilding FAISS indices for each query at the cost of extra search work.

        Args:
            index_name: Index name to search.
            query_embedding: Query embedding vector.
            valid_ids: Set of record IDs to include in the returned results.
            k: Maximum number of results to return after filtering.
            threshold: Optional minimum similarity threshold.

        Returns:
            A filtered list of `(record_id, similarity_score)` tuples.
        """
        # Search more than k to account for filtering
        raw_results = self.search(
            index_name, 
            query_embedding, 
            k=min(k * 3, self._indices[index_name].ntotal),
            threshold=threshold
        )
        
        # Filter
        filtered = [(rid, score) for rid, score in raw_results if rid in valid_ids]
        
        return filtered[:k]
    
    def save(self, index_name: Optional[str] = None) -> None:
        """Persist indices and ID maps to disk.

        Args:
            index_name: If provided, save only this index. Otherwise save all dirty indices.

        Returns:
            None.
        """
        indices_to_save = [index_name] if index_name else list(self._dirty)
        
        for name in indices_to_save:
            if name not in self._indices:
                logger.warning("Cannot save unknown index: %s", name)
                continue
                
            if len(self._id_maps[name]) != self._indices[name].ntotal:
                logger.error(
                    "Skipping save for %s: id map mismatch (ids=%s, vectors=%s)",
                    name,
                    len(self._id_maps[name]),
                    self._indices[name].ntotal,
                )
                continue
            
            try:
                faiss.write_index(
                    self._indices[name],
                    str(self._index_path(name))
                )
                np.save(
                    str(self._id_map_path(name)),
                    np.array(self._id_maps[name], dtype=object)
                )
                self._dirty.discard(name)
                logger.debug("Saved index '%s' with %d vectors", name, self._indices[name].ntotal)
            except Exception as e:
                logger.error("Failed to save index '%s': %s", name, e)
                raise
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes to any index.

        Returns:
            True if any index has been modified since the last save.
        """
        return len(self._dirty) > 0
    
    @contextmanager
    def batch_operation(self):
        """Context manager for batch operations with automatic persistence.
        
        Usage:
            with vector_store.batch_operation():
                vector_store.add("episodes", id1, emb1)
                vector_store.add("episodes", id2, emb2)
            # Automatically saves on exit
        
        Yields:
            Self for chaining operations.
        """
        try:
            yield self
        finally:
            if self._dirty:
                self.save()
    
    def get_record_id(self, index_name: str, faiss_id: int) -> Optional[str]:
        """Resolve a FAISS internal ID back to the corresponding record ID.

        Args:
            index_name: Index name to resolve within.
            faiss_id: FAISS internal numeric ID.

        Returns:
            The associated record ID if present; otherwise None.
        """
        if index_name not in self._id_maps:
            return None
        if faiss_id < 0 or faiss_id >= len(self._id_maps[index_name]):
            return None
        return self._id_maps[index_name][faiss_id]
    
    def get_statistics(self) -> dict:
        """Return basic index statistics for monitoring/debugging.

        Returns:
            A dictionary keyed by index name containing count, dimension, and dirty status.
        """
        return {
            name: {
                "count": self._indices[name].ntotal,
                "dimension": self.dimension,
                "has_unsaved_changes": name in self._dirty
            }
            for name in self._indices
        }

