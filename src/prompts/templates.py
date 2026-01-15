"""
LLM prompt templates for the episodic memory pipeline.

These prompts are designed to be:
- Clear and specific in their instructions
- Structured to produce parseable output
- Robust to varied input types

IMPORTANT: All prompts include explicit instructions to:
- Never return null for any field
- Use empty arrays [] instead of null for list fields
- Use empty strings "" instead of null for string fields
- Always include all required fields
"""

# Shared JSON output instructions to prevent null values
JSON_OUTPUT_RULES = """
CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] instead of null for lists
- Use empty strings "" instead of null for strings
- Use 0.5 as default for numeric scores if uncertain
- All fields must be present in the response
"""


class PromptTemplates:
    """Collection of prompt templates for memory operations."""

    # =========================================================================
    # INGESTION PROMPTS
    # =========================================================================

    MEMORY_WORTHINESS = """You are a memory curator for a personal AI assistant. Your job is to determine if a piece of text contains information worth remembering about the user.

INPUT TEXT:
{text}

CONTEXT (if available):
- Previous topic: {previous_topic}
- Conversation context: {context}

Analyze this text and determine:
1. Does it contain personal information, preferences, goals, facts, or experiences worth remembering?
2. Is it specific enough to be useful in future interactions?
3. Would forgetting this information negatively impact future assistance?

Things that ARE worth remembering:
- Personal facts (name, location, occupation, relationships)
- Preferences and opinions
- Goals and plans
- Experiences and events
- Learning progress or interests
- Important decisions or changes

Things that are NOT worth remembering:
- Generic greetings or small talk
- Requests for immediate tasks with no lasting relevance
- Hypothetical questions not about the user
- General knowledge questions
- Temporary states ("I'm tired right now")

CRITICAL JSON RULES:
- Never return null for any field
- All fields must be present
- Use false as default for is_memory_worthy if uncertain
- Use 0.5 as default for confidence if uncertain
- Use "none" as default for memory_type if not applicable

Respond in this exact JSON format:
{{
    "is_memory_worthy": false,
    "confidence": 0.5,
    "reason": "brief explanation of decision",
    "memory_type": "none"
}}

Example for memory-worthy input:
{{
    "is_memory_worthy": true,
    "confidence": 0.85,
    "reason": "Contains personal preference about learning goals",
    "memory_type": "preference"
}}"""

    EPISODE_EXTRACTION = """You are extracting structured episodic memory from user input.

INPUT TEXT:
{text}

TIMESTAMP: {timestamp}

Your task:
1. Extract the core memory content (what should be remembered)
2. Identify the memory type
3. Extract relevant topics and entities
4. Assess importance (0-1 scale)

Memory Types:
- episodic: An event or experience that happened ("I went to...", "Today I...")
- fact: A stable piece of information ("My name is...", "I work at...")
- goal: An intention or objective ("I want to...", "My goal is...")
- preference: A like/dislike/preference ("I prefer...", "I don't like...")
- reflection: Self-reflection or insight ("I realized that...", "I think I...")

Importance Guidelines:
- 0.9-1.0: Life events, major decisions, core identity facts
- 0.7-0.8: Significant preferences, active goals, key relationships
- 0.5-0.6: General interests, minor preferences, casual experiences
- 0.3-0.4: Routine events, temporary states
- 0.1-0.2: Barely relevant, might be useful someday

Extract topics as specific tags (e.g., "korean_language", "travel", "work_projects").
Extract entities as named things (people, places, organizations, specific items).

If time references are mentioned (e.g., "yesterday", "last week", "in March"), adjust the inferred event time accordingly.

CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] for topics and entities if none found
- Use "episodic" as default memory_type if uncertain
- Use 0.5 as default importance if uncertain
- Use "none" for occurred_at_offset if no time reference
- All fields must be present

Respond in this exact JSON format:
{{
    "content": "cleaned, clear statement of what to remember",
    "memory_type": "episodic",
    "topics": [],
    "entities": [],
    "importance": 0.5,
    "occurred_at_offset": "none"
}}

Example with extracted values:
{{
    "content": "Started learning Korean using Duolingo app",
    "memory_type": "episodic",
    "topics": ["korean_language", "learning", "duolingo"],
    "entities": ["Duolingo"],
    "importance": 0.7,
    "occurred_at_offset": "none"
}}"""

    # =========================================================================
    # CONSOLIDATION PROMPTS
    # =========================================================================

    SUMMARIZATION = """You are creating a narrative summary of recent episodic memories for a personal AI assistant.

TOPIC: {topic}
TIME PERIOD: {time_start} to {time_end}

EPISODES TO SUMMARIZE:
{episodes}

Your task:
1. Create a coherent narrative summary that captures the key developments
2. Identify the 2-4 most significant events
3. Note any patterns, progress, or changes
4. Preserve important details while reducing redundancy

The summary should:
- Read like a brief journal entry or status update
- Be written in third person ("The user...")
- Highlight progression over time if applicable
- Be 2-4 paragraphs maximum

CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] for key_events, themes, notable_changes if none found
- Always provide a summary string (never null)
- All fields must be present

Respond in this exact JSON format:
{{
    "summary": "The narrative summary text...",
    "key_events": [],
    "themes": [],
    "notable_changes": []
}}

Example with content:
{{
    "summary": "The user has been actively learning Korean over the past week. They completed several Duolingo lessons and started watching Korean dramas for immersion practice.",
    "key_events": ["Completed 5 Duolingo lessons", "Started watching first K-drama"],
    "themes": ["language_learning", "self_improvement"],
    "notable_changes": ["Shifted from app-only to multimedia learning approach"]
}}"""

    FACT_EXTRACTION = """You are extracting stable facts from episodic memories for a personal AI assistant.

TOPIC: {topic}

EPISODES:
{episodes}

EXISTING FACTS (may need updating):
{existing_facts}

Your task:
1. Extract facts that are stable and likely to remain true
2. Identify if any existing facts need updating or contradicting
3. Assign confidence based on evidence strength

Fact Categories:
- personal: About the user's identity
- preference: User preferences
- relationship: People and connections
- knowledge: Things the user knows
- context: Situational context
- goal: Long-term goals

For each fact, determine:
- Is this new information?
- Does it update an existing fact?
- Does it contradict an existing fact?
- How confident are we? (based on clarity, repetition, recency)

CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] for new_facts, updated_facts, contradicted_facts if none found
- Each fact object must have all required fields
- Use empty string "" instead of null for string fields
- Use 0.8 as default confidence if uncertain
- All fields must be present

Respond in this exact JSON format (empty arrays are valid):
{{
    "new_facts": [],
    "updated_facts": [],
    "contradicted_facts": []
}}

Example with extracted facts:
{{
    "new_facts": [
        {{
            "content": "User is learning Korean",
            "category": "goal",
            "confidence": 0.9,
            "source_hint": "Episode about starting Duolingo"
        }}
    ],
    "updated_facts": [],
    "contradicted_facts": []
}}"""

    # =========================================================================
    # RETRIEVAL PROMPTS
    # =========================================================================

    QUERY_ANALYSIS = """Analyze this query to determine the best retrieval strategy.

QUERY: {query}

USER CONTEXT:
- Known topics: {known_topics}
- Recent activity: {recent_activity}

Determine:
1. Query type: semantic (concept/meaning-based) or narrative (story/timeline-based)
2. Time relevance: all_time, recent, specific_period
3. Key concepts to search for
4. Any topic filters to apply

CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] for search_concepts and topic_filters if none
- Use empty strings "" in time_filter instead of null
- Use "semantic" as default query_type if uncertain
- Use "all_time" as default time_relevance if uncertain
- All fields must be present

Respond in this exact JSON format:
{{
    "query_type": "semantic",
    "time_relevance": "all_time",
    "time_filter": {{"since": "", "until": ""}},
    "search_concepts": [],
    "topic_filters": [],
    "reformulated_query": "optimized query for vector search"
}}

Example with values:
{{
    "query_type": "narrative",
    "time_relevance": "recent",
    "time_filter": {{"since": "2024-01-01", "until": ""}},
    "search_concepts": ["korean", "language", "learning"],
    "topic_filters": ["language_learning"],
    "reformulated_query": "Korean language learning progress and activities"
}}"""

    ANSWER_SYNTHESIS = """You are synthesizing an answer from retrieved memories.

ORIGINAL QUERY: {query}

RETRIEVED MEMORIES:

SUMMARIES:
{summaries}

RELEVANT FACTS:
{facts}

SUPPORTING EPISODES:
{episodes}

Your task:
1. Synthesize a coherent answer to the query
2. Reference specific memories when relevant
3. Acknowledge uncertainty if information is incomplete
4. Distinguish between facts and episodes

Guidelines:
- Prefer recent information over old
- Prefer high-confidence facts over single episodes
- If there's a journey/progression, tell it chronologically
- If contradictory information exists, acknowledge it

CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] for key_sources and gaps if none
- Always provide an answer string (never null)
- Use 0.5 as default confidence if uncertain
- All fields must be present

Respond in this exact JSON format:
{{
    "answer": "Your synthesized answer...",
    "confidence": 0.5,
    "key_sources": [],
    "gaps": []
}}

Example with content:
{{
    "answer": "Based on your memories, you have been learning Korean for about 3 weeks, primarily using Duolingo.",
    "confidence": 0.85,
    "key_sources": ["Episode from Jan 15 about starting Duolingo", "Fact: User is learning Korean"],
    "gaps": ["No information about current proficiency level"]
}}"""

    NARRATIVE_SYNTHESIS = """You are reconstructing a narrative from episodic memories.

TOPIC/SUBJECT: {topic}
QUERY: {query}

EPISODES (chronologically ordered):
{episodes}

RELEVANT FACTS:
{facts}

RELATED SUMMARIES:
{summaries}

Your task:
Reconstruct the narrative journey around this topic. Tell the story of what happened,
capturing the progression, key moments, and how things evolved over time.

The narrative should:
- Flow chronologically
- Highlight turning points and key decisions
- Connect related events
- Feel like recounting a memory, not reading a database

CRITICAL JSON RULES:
- Never return null for any field
- Use empty arrays [] for timeline and key_moments if none
- Always provide narrative and current_status strings (never null)
- Use empty string "" if current_status is unknown
- All fields must be present

Respond in this exact JSON format:
{{
    "narrative": "The flowing narrative text...",
    "timeline": [],
    "key_moments": [],
    "current_status": ""
}}

Example with content:
{{
    "narrative": "The user began their Korean language learning journey in early January. They started with Duolingo and quickly established a daily practice routine.",
    "timeline": [
        {{"date": "2024-01-15", "event": "Started using Duolingo for Korean"}},
        {{"date": "2024-01-20", "event": "Completed first lesson module"}}
    ],
    "key_moments": ["Decision to learn Korean", "First successful conversation practice"],
    "current_status": "Actively learning, completing daily lessons"
}}"""

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @classmethod
    def format_episodes_for_prompt(cls, episodes: list, max_chars: int = 4000) -> str:
        """Format episodes for inclusion in prompts.

        Args:
            episodes: Episodes (objects or dicts) to format.
            max_chars: Maximum character budget for the formatted output.

        Returns:
            A newline-delimited string suitable for prompt inclusion.
        """
        lines = []
        total_chars = 0

        for i, ep in enumerate(episodes):
            # Handle both Episode objects and dicts
            if hasattr(ep, "occurred_at"):
                timestamp = ep.occurred_at.strftime("%Y-%m-%d %H:%M")
                content = ep.content
                ep_id = ep.id[:8]
            else:
                timestamp = ep.get("occurred_at", "unknown")
                content = ep.get("content", str(ep))
                ep_id = ep.get("id", str(i))[:8]

            line = f"[{timestamp}] ({ep_id}): {content}"

            if total_chars + len(line) > max_chars:
                lines.append(f"... and {len(episodes) - i} more episodes")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    @classmethod
    def format_facts_for_prompt(cls, facts: list, max_chars: int = 2000) -> str:
        """Format facts for inclusion in prompts.

        Args:
            facts: Facts (objects or dicts) to format.
            max_chars: Maximum character budget for the formatted output.

        Returns:
            A newline-delimited string suitable for prompt inclusion.
        """
        lines = []
        total_chars = 0

        for i, fact in enumerate(facts):
            if hasattr(fact, "content"):
                content = fact.content
                category = fact.category
                fact_id = fact.id[:8]
                confidence = fact.confidence
            else:
                content = fact.get("content", str(fact))
                category = fact.get("category", "unknown")
                fact_id = fact.get("id", str(i))[:8]
                confidence = fact.get("confidence", 1.0)

            line = f"[{fact_id}] ({category}, conf={confidence:.1f}): {content}"

            if total_chars + len(line) > max_chars:
                lines.append(f"... and {len(facts) - i} more facts")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines) if lines else "No facts available."

    @classmethod
    def format_summaries_for_prompt(cls, summaries: list, max_chars: int = 3000) -> str:
        """Format summaries for inclusion in prompts.

        Args:
            summaries: Summaries (objects or dicts) to format.
            max_chars: Maximum character budget for the formatted output.

        Returns:
            A newline-delimited string suitable for prompt inclusion.
        """
        lines = []
        total_chars = 0

        for i, summary in enumerate(summaries):
            if hasattr(summary, "content"):
                content = summary.content
                topic = summary.topic
                time_start = summary.time_start.strftime("%Y-%m-%d")
                time_end = summary.time_end.strftime("%Y-%m-%d")
            else:
                content = summary.get("content", str(summary))
                topic = summary.get("topic", "unknown")
                time_start = summary.get("time_start", "?")
                time_end = summary.get("time_end", "?")

            line = f"[{topic}: {time_start} to {time_end}]\n{content}\n"

            if total_chars + len(line) > max_chars:
                lines.append(f"... and {len(summaries) - i} more summaries")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines) if lines else "No summaries available."
