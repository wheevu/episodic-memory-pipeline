"""
Fact extractor - extracts stable facts from episodes.

Identifies and manages semantic memory (facts) derived from episodic memories.
Handles fact lifecycle: creation, updates, contradictions.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..llm import LLMProvider
from ..models import Episode, Fact, FactCategory
from ..prompts import PromptTemplates
from ..utils import as_dict, as_float, as_list, as_str


@dataclass
class FactExtractionResult:
    """Represents the outcome of extracting facts from a set of episodes.

    Attributes:
        new_facts: Facts newly created from the provided episodes.
        updated_facts: Pairs of (old_fact_id, new_fact) representing updates.
        contradicted_fact_ids: IDs of existing facts deemed contradicted.
        source_episode_ids: Episode IDs that served as evidence/provenance.
    """

    new_facts: list[Fact]
    updated_facts: list[tuple[str, Fact]]  # (old_id, new_fact)
    contradicted_fact_ids: list[str]
    source_episode_ids: list[str]


class FactExtractor:
    """
    Extracts stable facts from episodic memories.

    Design principles:
    - Facts are derived from episodes, not created directly
    - Confidence reflects evidence strength
    - Facts can be updated or contradicted over time
    - Provenance is always maintained
    """

    def __init__(self, llm: LLMProvider) -> None:
        """Initialize the fact extractor.

        Args:
            llm: LLM provider used to perform structured fact extraction.
        """
        self.llm = llm

    def extract_facts(
        self,
        episodes: list[Episode],
        topic: str,
        existing_facts: Optional[list[Fact]] = None,
    ) -> FactExtractionResult:
        """Extract stable facts from episodes, considering existing facts.

        Args:
            episodes: Episodes to extract facts from.
            topic: Topic context used in the extraction prompt.
            existing_facts: Currently known facts (may be updated/contradicted).

        Returns:
            A `FactExtractionResult` containing new, updated, and contradicted facts.

        Raises:
            json.JSONDecodeError: If the LLM returns invalid JSON (caught and handled).
            KeyError: If expected keys are missing in the response (caught and handled).
        """
        existing_facts = existing_facts or []

        if not episodes:
            return FactExtractionResult(
                new_facts=[], updated_facts=[], contradicted_fact_ids=[], source_episode_ids=[]
            )

        # Format for prompt
        episode_text = PromptTemplates.format_episodes_for_prompt(episodes)
        facts_text = PromptTemplates.format_facts_for_prompt(existing_facts)

        prompt = PromptTemplates.FACT_EXTRACTION.format(
            topic=topic, episodes=episode_text, existing_facts=facts_text
        )

        try:
            response = self.llm.complete(prompt)
            result = json.loads(response)

            return self._process_extraction_result(result, topic, episodes, existing_facts)

        except (json.JSONDecodeError, KeyError):
            # Fallback: no facts extracted
            return FactExtractionResult(
                new_facts=[],
                updated_facts=[],
                contradicted_fact_ids=[],
                source_episode_ids=[ep.id for ep in episodes],
            )

    def _process_extraction_result(
        self, result: dict, topic: str, episodes: list[Episode], existing_facts: list[Fact]
    ) -> FactExtractionResult:
        """Convert an LLM extraction payload into structured `Fact` objects.

        Args:
            result: Parsed JSON payload from the LLM.
            topic: Topic label applied to created/updated facts.
            episodes: Episodes used as evidence for provenance.
            existing_facts: Existing facts used for update/contradiction matching.

        Returns:
            A `FactExtractionResult` with normalized `Fact` objects.
        """

        episode_ids = [ep.id for ep in episodes]
        new_facts = []
        updated_facts = []
        contradicted_ids = []

        # Create lookup for existing facts
        # The prompt asks the LLM to refer to existing facts by a short, human-friendly
        # identifier prefix (first 8 chars) rather than a full UUID.
        existing_by_id = {f.id[:8]: f for f in existing_facts}

        # Sanitize list fields using centralized helpers
        raw_new_facts = as_list(result.get("new_facts"))
        raw_updated_facts = as_list(result.get("updated_facts"))
        raw_contradicted = as_list(result.get("contradicted_facts"))

        # Process new facts
        for raw_fact in raw_new_facts:
            raw_fact = as_dict(raw_fact)
            if not raw_fact:
                continue

            # Sanitize individual fact fields
            content = as_str(raw_fact.get("content"))
            if not content:
                continue

            category_str = as_str(raw_fact.get("category"), default="personal")
            try:
                category = FactCategory(category_str)
            except ValueError:
                category = FactCategory.PERSONAL

            confidence = as_float(raw_fact.get("confidence"), default=0.8)

            fact = Fact(
                content=content,
                category=category,
                topic=topic,
                confidence=confidence,
                valid_from=datetime.now(timezone.utc),
            )
            new_facts.append(fact)

        # Process updated facts
        for raw_update in raw_updated_facts:
            raw_update = as_dict(raw_update)
            if not raw_update:
                continue

            old_id = as_str(raw_update.get("existing_fact_id"))
            if old_id in existing_by_id:
                old_fact = existing_by_id[old_id]
                new_content = (
                    as_str(raw_update.get("new_content"), default=old_fact.content)
                    or old_fact.content
                )

                new_fact = Fact(
                    content=new_content,
                    category=old_fact.category,
                    topic=topic,
                    confidence=max(0.7, old_fact.confidence),  # Slight confidence boost
                    valid_from=datetime.now(timezone.utc),
                )
                updated_facts.append((old_fact.id, new_fact))

        # Process contradicted facts
        for raw_contradict in raw_contradicted:
            raw_contradict = as_dict(raw_contradict)
            if not raw_contradict:
                continue

            old_id = as_str(raw_contradict.get("existing_fact_id"))
            if old_id in existing_by_id:
                contradicted_ids.append(existing_by_id[old_id].id)

        return FactExtractionResult(
            new_facts=new_facts,
            updated_facts=updated_facts,
            contradicted_fact_ids=contradicted_ids,
            source_episode_ids=episode_ids,
        )

    def merge_similar_facts(self, facts: list[Fact]) -> list[Fact]:
        """Merge semantically similar facts using simple heuristics.

        Args:
            facts: Facts to merge.

        Returns:
            A list of facts where highly similar entries have been combined.
        """
        if len(facts) <= 1:
            return facts

        merged = []
        used = set()

        for i, fact1 in enumerate(facts):
            if i in used:
                continue

            # Find similar facts
            similar = [fact1]
            for j, fact2 in enumerate(facts[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if self._are_similar(fact1, fact2):
                    similar.append(fact2)
                    used.add(j)

            # Merge if multiple
            if len(similar) > 1:
                merged_fact = self._merge_facts(similar)
                merged.append(merged_fact)
            else:
                merged.append(fact1)

            used.add(i)

        return merged

    def _are_similar(self, fact1: Fact, fact2: Fact) -> bool:
        """Return True if two facts are similar enough to merge.

        Args:
            fact1: First fact.
            fact2: Second fact.

        Returns:
            True if the facts should be considered merge candidates.
        """
        # Same topic and category
        if fact1.topic != fact2.topic or fact1.category != fact2.category:
            return False

        # Word overlap heuristic
        words1 = set(fact1.content.lower().split())
        words2 = set(fact2.content.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard = intersection / union if union > 0 else 0
        # A higher threshold biases toward precision over recall; we only merge when
        # overlap is strong to avoid collapsing distinct facts.
        return jaccard > 0.5

    def _merge_facts(self, facts: list[Fact]) -> Fact:
        """Merge multiple similar facts into a single fact.

        Args:
            facts: Facts presumed similar by `_are_similar`.

        Returns:
            A merged `Fact` representing the strongest candidate.
        """
        # Use the most recent, highest confidence fact as base
        facts = sorted(facts, key=lambda f: (f.confidence, f.created_at), reverse=True)
        best = facts[0]

        # Boost confidence based on reinforcement
        new_confidence = min(1.0, best.confidence + 0.1 * (len(facts) - 1))

        return Fact(
            content=best.content,
            category=best.category,
            topic=best.topic,
            confidence=new_confidence,
            valid_from=min(f.created_at for f in facts),
            entities=best.entities,
        )
