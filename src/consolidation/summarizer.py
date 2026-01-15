"""
Summarizer - generates narrative summaries from episodes.

Consolidates multiple episodes into coherent narrative summaries,
preserving key events while reducing redundancy.
"""

import json
from dataclasses import dataclass
from typing import Optional

from ..llm import LLMProvider
from ..models import Episode, Summary
from ..prompts import PromptTemplates
from ..utils import as_list, as_str


@dataclass
class SummarizationResult:
    """Represents the outcome of summarizing a set of episodes.

    Attributes:
        summary: Generated summary object.
        key_episode_ids: IDs of episodes identified as key contributors.
        themes: Themes extracted from the episode set.
        notable_changes: Notable changes detected across the time span.
    """

    summary: Summary
    key_episode_ids: list[str]
    themes: list[str]
    notable_changes: list[str]


class Summarizer:
    """
    Generates narrative summaries from episode collections.

    Design principles:
    - Summaries should read like journal entries
    - Key events are explicitly identified for provenance
    - Temporal ordering is preserved in the narrative
    - Summaries can be hierarchical (weekly → monthly)
    """

    def __init__(self, llm: LLMProvider, max_episodes_per_summary: int = 20) -> None:
        """Initialize the summarizer.

        Args:
            llm: LLM provider used for summary generation.
            max_episodes_per_summary: Maximum episodes to include in a single summary.
        """
        self.llm = llm
        self.max_episodes = max_episodes_per_summary

    def summarize(
        self,
        episodes: list[Episode],
        topic: str,
        existing_summary: Optional[Summary] = None,
    ) -> SummarizationResult:
        """Generate a narrative summary from a set of topic-filtered episodes.

        Args:
            episodes: Episodes to summarize (expected pre-filtered by topic).
            topic: Topic being summarized.
            existing_summary: Previous summary to update (currently unused).

        Returns:
            A `SummarizationResult` containing the summary and extraction metadata.

        Raises:
            ValueError: If `episodes` is empty.
        """
        if not episodes:
            raise ValueError("Cannot summarize empty episode list")

        # Sort by time
        episodes = sorted(episodes, key=lambda e: e.occurred_at)

        # Limit to max episodes (take most recent if exceeding)
        if len(episodes) > self.max_episodes:
            episodes = episodes[-self.max_episodes :]

        # Determine time range
        time_start = episodes[0].occurred_at
        time_end = episodes[-1].occurred_at

        # Format episodes for prompt
        episode_text = PromptTemplates.format_episodes_for_prompt(episodes)

        prompt = PromptTemplates.SUMMARIZATION.format(
            topic=topic,
            time_start=time_start.strftime("%Y-%m-%d"),
            time_end=time_end.strftime("%Y-%m-%d"),
            episodes=episode_text,
        )

        try:
            response = self.llm.complete(prompt)
            result = json.loads(response)

            # Sanitize all LLM output fields using centralized helpers
            key_events = as_list(result.get("key_events"))[:5]  # Limit to 5
            themes = as_list(result.get("themes"))
            notable_changes = as_list(result.get("notable_changes"))
            summary_content = (
                as_str(result.get("summary"), default="Summary generation failed.")
                or "Summary generation failed."
            )

            # Create summary with sanitized values
            summary = Summary(
                content=summary_content,
                topic=topic,
                time_start=time_start,
                time_end=time_end,
                episode_count=len(episodes),
                key_events=key_events,
                summary_level=1,  # Default to weekly level
            )

            # Identify key episodes (ones that contributed key events)
            key_episode_ids = self._identify_key_episodes(episodes, key_events)

            return SummarizationResult(
                summary=summary,
                key_episode_ids=key_episode_ids,
                themes=themes,
                notable_changes=notable_changes,
            )

        except (json.JSONDecodeError, KeyError):
            # Fallback: create basic summary
            summary = Summary(
                content=f"Summary of {len(episodes)} episodes about {topic} from {time_start.date()} to {time_end.date()}.",
                topic=topic,
                time_start=time_start,
                time_end=time_end,
                episode_count=len(episodes),
                key_events=[],
                summary_level=1,
            )

            return SummarizationResult(
                summary=summary, key_episode_ids=[], themes=[], notable_changes=[]
            )

    def _identify_key_episodes(self, episodes: list[Episode], key_events: list[str]) -> list[str]:
        """Identify which episodes correspond to extracted key events.

        Uses a simple word-overlap heuristic to keep this deterministic and cheap.

        Args:
            episodes: Episodes that were summarized.
            key_events: Key event strings extracted by the LLM.

        Returns:
            A list of episode IDs aligned (roughly) to the key events list.
        """
        if not key_events:
            return []

        key_ids = []
        key_events_lower = [e.lower() for e in key_events]

        for episode in episodes:
            content_lower = episode.content.lower()
            for key_event in key_events_lower:
                # Check if key event mentions are in episode
                key_words = set(key_event.split())
                episode_words = set(content_lower.split())
                overlap = len(key_words & episode_words) / len(key_words)
                if overlap > 0.5:  # >50% word overlap
                    key_ids.append(episode.id)
                    break

        return key_ids[: len(key_events)]  # Limit to number of key events

    def create_higher_level_summary(
        self, summaries: list[Summary], topic: str, level: int = 2
    ) -> Summary:
        """Create a higher-level summary by aggregating lower-level summaries.

        Args:
            summaries: Lower-level summaries to aggregate.
            topic: Topic being summarized.
            level: Summary level to assign (e.g., 2 for monthly).

        Returns:
            A synthesized `Summary` across the provided time span.

        Raises:
            ValueError: If `summaries` is empty.
        """
        if not summaries:
            raise ValueError("Cannot create higher-level summary from empty list")

        # Sort by time
        summaries = sorted(summaries, key=lambda s: s.time_start)

        time_start = summaries[0].time_start
        time_end = summaries[-1].time_end
        total_episodes = sum(s.episode_count for s in summaries)

        # Combine summaries for prompt
        combined_text = "\n\n".join(
            [
                f"[{s.time_start.strftime('%Y-%m-%d')} to {s.time_end.strftime('%Y-%m-%d')}]\n{s.content}"
                for s in summaries
            ]
        )

        prompt = f"""Create a higher-level summary that synthesizes these period summaries into a cohesive narrative.

TOPIC: {topic}
TIME PERIOD: {time_start.strftime("%Y-%m-%d")} to {time_end.strftime("%Y-%m-%d")}

PERIOD SUMMARIES:
{combined_text}

Create a cohesive narrative that:
1. Captures the overall arc and progression
2. Identifies major themes across periods
3. Notes significant changes or developments
4. Maintains key details while reducing redundancy

Respond in JSON format:
{{
    "summary": "The synthesized narrative...",
    "key_events": ["Most significant events across all periods"],
    "overall_themes": ["Major recurring themes"]
}}"""

        try:
            response = self.llm.complete(prompt)
            result = json.loads(response)

            # Sanitize LLM output fields
            key_events = as_list(result.get("key_events"))
            summary_content = (
                as_str(result.get("summary"), default="Higher-level summary generation failed.")
                or "Higher-level summary generation failed."
            )

            return Summary(
                content=summary_content,
                topic=topic,
                time_start=time_start,
                time_end=time_end,
                episode_count=total_episodes,
                key_events=key_events,
                summary_level=level,
            )
        except (json.JSONDecodeError, KeyError):
            return Summary(
                content=f"Summary of {topic} from {time_start.date()} to {time_end.date()} covering {len(summaries)} periods.",
                topic=topic,
                time_start=time_start,
                time_end=time_end,
                episode_count=total_episodes,
                key_events=[],
                summary_level=level,
            )
