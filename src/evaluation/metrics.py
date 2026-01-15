"""
Evaluation metrics for the episodic memory pipeline.

Implements three core metrics:
1. Retrieval Precision@K - measures retrieval quality
2. Fact Conflict Rate - measures fact consistency
3. Consolidation Compression Ratio - measures summarization efficiency
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RetrievalPrecisionResult:
    """Result of retrieval precision evaluation."""

    precision_at_k: float
    k: int
    relevant_found: int
    total_expected: int
    retrieved_ids: list[str]
    expected_ids: list[str]

    @property
    def recall(self) -> float:
        """Compute recall = relevant_found / total_expected.

        Returns:
            Recall value in the range [0.0, 1.0].
        """
        if self.total_expected == 0:
            return 0.0
        return self.relevant_found / self.total_expected

    @property
    def f1(self) -> float:
        """Compute F1 score (harmonic mean of precision and recall).

        Returns:
            F1 value in the range [0.0, 1.0].
        """
        if self.precision_at_k + self.recall == 0:
            return 0.0
        return 2 * (self.precision_at_k * self.recall) / (self.precision_at_k + self.recall)


@dataclass
class FactConflictResult:
    """Result of fact conflict rate evaluation."""

    conflict_rate: float
    total_facts: int
    conflicting_facts: int
    conflict_pairs: list[tuple[str, str, str]]  # (fact1_id, fact2_id, reason)

    @property
    def consistency_rate(self) -> float:
        """Compute consistency rate = 1 - conflict_rate.

        Returns:
            Consistency rate in the range [0.0, 1.0].
        """
        return 1.0 - self.conflict_rate


@dataclass
class CompressionResult:
    """Result of consolidation compression evaluation."""

    compression_ratio: float
    source_tokens: int
    summary_tokens: int
    episode_count: int
    summary_count: int

    @property
    def tokens_per_episode(self) -> float:
        """Compute average tokens per source episode.

        Returns:
            Average tokens per source episode, or 0.0 if there are no episodes.
        """
        if self.episode_count == 0:
            return 0.0
        return self.source_tokens / self.episode_count

    @property
    def tokens_per_summary(self) -> float:
        """Compute average tokens per summary.

        Returns:
            Average tokens per summary, or 0.0 if there are no summaries.
        """
        if self.summary_count == 0:
            return 0.0
        return self.summary_tokens / self.summary_count


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    retrieval_precision: Optional[RetrievalPrecisionResult] = None
    fact_conflict: Optional[FactConflictResult] = None
    compression: Optional[CompressionResult] = None

    # Additional metadata
    scenario_name: str = ""
    timestamp: str = ""
    episode_count: int = 0
    fact_count: int = 0
    summary_count: int = 0

    # Provider status (for showing SKIPPED warnings)
    using_mock_embeddings: bool = False
    using_mock_llm: bool = False

    def to_dict(self) -> dict:
        """Convert metrics to a JSON-serializable dictionary.

        Returns:
            A dictionary suitable for JSON serialization.
        """
        result = {
            "scenario": self.scenario_name,
            "timestamp": self.timestamp,
            "counts": {
                "episodes": self.episode_count,
                "facts": self.fact_count,
                "summaries": self.summary_count,
            },
        }

        if self.retrieval_precision:
            result["retrieval"] = {
                "precision_at_k": self.retrieval_precision.precision_at_k,
                "k": self.retrieval_precision.k,
                "recall": self.retrieval_precision.recall,
                "f1": self.retrieval_precision.f1,
                "relevant_found": self.retrieval_precision.relevant_found,
                "total_expected": self.retrieval_precision.total_expected,
            }

        if self.fact_conflict:
            result["fact_consistency"] = {
                "conflict_rate": self.fact_conflict.conflict_rate,
                "consistency_rate": self.fact_conflict.consistency_rate,
                "total_facts": self.fact_conflict.total_facts,
                "conflicting_facts": self.fact_conflict.conflicting_facts,
            }

        if self.compression:
            result["compression"] = {
                "ratio": self.compression.compression_ratio,
                "source_tokens": self.compression.source_tokens,
                "summary_tokens": self.compression.summary_tokens,
            }

        return result


class RetrievalPrecisionMetric:
    """
    Measures Retrieval Precision@K.

    Precision@K = (# relevant episodes in top-k results) / k

    A relevant episode is one whose ID is in the expected set.
    """

    def __init__(self, k: int = 5) -> None:
        """
        Initialize metric.

        Args:
            k: Number of top results to consider

        Returns:
            None.
        """
        self.k = k

    def evaluate(
        self,
        retrieved_episode_ids: list[str],
        expected_episode_ids: list[str],
    ) -> RetrievalPrecisionResult:
        """
        Compute precision@k.

        Args:
            retrieved_episode_ids: IDs of retrieved episodes (in rank order)
            expected_episode_ids: IDs of episodes that should be retrieved

        Returns:
            RetrievalPrecisionResult with metrics
        """
        expected_set = set(expected_episode_ids)
        top_k = retrieved_episode_ids[: self.k]

        relevant_found = sum(1 for eid in top_k if eid in expected_set)
        precision = relevant_found / self.k if self.k > 0 else 0.0

        return RetrievalPrecisionResult(
            precision_at_k=precision,
            k=self.k,
            relevant_found=relevant_found,
            total_expected=len(expected_episode_ids),
            retrieved_ids=top_k,
            expected_ids=expected_episode_ids,
        )

    def evaluate_multiple(
        self,
        query_results: list[tuple[list[str], list[str]]],
    ) -> RetrievalPrecisionResult:
        """
        Compute mean precision@k across multiple queries.

        Args:
            query_results: List of (retrieved_ids, expected_ids) tuples

        Returns:
            Aggregated RetrievalPrecisionResult
        """
        if not query_results:
            return RetrievalPrecisionResult(
                precision_at_k=0.0,
                k=self.k,
                relevant_found=0,
                total_expected=0,
                retrieved_ids=[],
                expected_ids=[],
            )

        total_precision = 0.0
        total_relevant = 0
        total_expected = 0

        for retrieved, expected in query_results:
            result = self.evaluate(retrieved, expected)
            total_precision += result.precision_at_k
            total_relevant += result.relevant_found
            total_expected += result.total_expected

        mean_precision = total_precision / len(query_results)

        return RetrievalPrecisionResult(
            precision_at_k=mean_precision,
            k=self.k,
            relevant_found=total_relevant,
            total_expected=total_expected,
            retrieved_ids=[],  # Not meaningful for aggregate
            expected_ids=[],
        )


class FactConflictRateMetric:
    """
    Measures Fact Conflict Rate.

    A conflict is defined as two facts about the same entity/attribute
    with different values.

    Conflict Rate = (# facts with at least one conflict) / (total facts)
    """

    def __init__(self) -> None:
        """Initialize the metric.

        Returns:
            None.
        """

    def evaluate(self, facts: list) -> FactConflictResult:
        """
        Compute fact conflict rate.

        Args:
            facts: List of Fact objects from the database

        Returns:
            FactConflictResult with conflict analysis
        """
        if not facts:
            return FactConflictResult(
                conflict_rate=0.0,
                total_facts=0,
                conflicting_facts=0,
                conflict_pairs=[],
            )

        # Group facts by entity-attribute signature
        fact_groups = defaultdict(list)

        for fact in facts:
            # Extract entity-attribute signature from fact content
            signature = self._extract_signature(fact)
            if signature:
                fact_groups[signature].append(fact)

        # Find conflicts (groups with multiple different values)
        conflict_pairs = []
        conflicting_fact_ids = set()

        for signature, group in fact_groups.items():
            if len(group) < 2:
                continue

            # Compare facts within group for value differences
            for i, fact1 in enumerate(group):
                for fact2 in group[i + 1 :]:
                    if self._are_conflicting(fact1, fact2):
                        conflict_pairs.append(
                            (
                                fact1.id,
                                fact2.id,
                                f"Same attribute '{signature}' with different values",
                            )
                        )
                        conflicting_fact_ids.add(fact1.id)
                        conflicting_fact_ids.add(fact2.id)

        conflict_rate = len(conflicting_fact_ids) / len(facts) if facts else 0.0

        return FactConflictResult(
            conflict_rate=conflict_rate,
            total_facts=len(facts),
            conflicting_facts=len(conflicting_fact_ids),
            conflict_pairs=conflict_pairs,
        )

    def _extract_signature(self, fact: Any) -> Optional[str]:
        """
        Extract entity-attribute signature from a fact.

        Examples:
        - "User's name is John" -> "user:name"
        - "User lives in NYC" -> "user:location"
        - "User is learning Korean" -> "user:learning"

        Args:
            fact: Fact-like object with a `content` attribute and optional `topic`/`category`.

        Returns:
            Signature string used for grouping facts, or None if no signature is available.
        """
        content = fact.content.lower()

        # Common patterns for entity-attribute extraction
        patterns = [
            (r"user(?:'s)?\s+(\w+)\s+is", "user"),
            (r"user\s+(\w+s?)\s+(?:in|at|to)", "user"),
            (r"user\s+is\s+(\w+ing)", "user"),
            (r"user\s+(?:prefers?|likes?|wants?)\s+(\w+)", "user"),
        ]

        for pattern, entity in patterns:
            match = re.search(pattern, content)
            if match:
                attribute = match.group(1)
                return f"{entity}:{attribute}"

        # Fallback: use topic + category as signature
        topic = getattr(fact, "topic", "unknown")
        category = getattr(fact, "category", "unknown")
        return f"{topic}:{category}"

    def _are_conflicting(self, fact1: Any, fact2: Any) -> bool:
        """
        Determine if two facts conflict.

        Two facts conflict if they're about the same thing but state
        different values.

        Args:
            fact1: First fact-like object.
            fact2: Second fact-like object.

        Returns:
            True if the facts are considered conflicting; otherwise False.
        """
        # If one supersedes the other, not a conflict (it's an update)
        if getattr(fact1, "superseded_by", None) == fact2.id:
            return False
        if getattr(fact2, "superseded_by", None) == fact1.id:
            return False

        # Check if contents are substantially different
        content1 = fact1.content.lower()
        content2 = fact2.content.lower()

        # If contents are very similar, not a conflict
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 or not words2:
            return False

        jaccard = len(words1 & words2) / len(words1 | words2)

        # If > 80% similar, treat as same fact (not conflict)
        # If < 30% similar and same signature, likely conflict
        return jaccard < 0.3


class ConsolidationCompressionMetric:
    """
    Measures Consolidation Compression Ratio.

    Compression Ratio = (summary tokens) / (source episode tokens)

    Lower ratio = better compression (more information density)
    Typical good range: 0.1-0.3 (70-90% compression)
    """

    def __init__(self) -> None:
        """Initialize the metric.

        Returns:
            None.
        """

    def evaluate(
        self,
        episodes: list,
        summaries: list,
    ) -> CompressionResult:
        """
        Compute compression ratio.

        Args:
            episodes: List of Episode objects (sources)
            summaries: List of Summary objects (compressed)

        Returns:
            CompressionResult with compression analysis
        """
        if not episodes:
            return CompressionResult(
                compression_ratio=0.0,
                source_tokens=0,
                summary_tokens=0,
                episode_count=0,
                summary_count=len(summaries),
            )

        # Count tokens in source episodes
        source_tokens = sum(self._count_tokens(ep.content) for ep in episodes)

        # Count tokens in summaries
        summary_tokens = sum(self._count_tokens(s.content) for s in summaries) if summaries else 0

        # Compute ratio
        ratio = summary_tokens / source_tokens if source_tokens > 0 else 0.0

        return CompressionResult(
            compression_ratio=ratio,
            source_tokens=source_tokens,
            summary_tokens=summary_tokens,
            episode_count=len(episodes),
            summary_count=len(summaries),
        )

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count using simple word splitting.

        This is a rough approximation. For exact counts, use a
        proper tokenizer (tiktoken for GPT, etc.)

        Approximation: 1 token ≈ 0.75 words (for English)

        Args:
            text: Input text to approximate token count for.

        Returns:
            Approximate token count as an integer.
        """
        if not text:
            return 0

        # Split on whitespace and punctuation
        words = re.findall(r"\b\w+\b", text)

        # Approximate tokens (English averages ~1.3 tokens per word)
        return int(len(words) * 1.3)
