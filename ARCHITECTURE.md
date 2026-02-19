# Episodic Memory Pipeline - Architecture & Evaluation

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                │
│                    CLI / API / Interactive Session                         │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              ┌─────▼─────┐              ┌──────▼──────┐
              │ INGESTION │              │  RETRIEVAL  │
              │ PIPELINE  │              │   ENGINE    │
              └─────┬─────┘              └──────┬──────┘
                    │                           │
    ┌───────────────┼───────────────┐          │
    │               │               │          │
┌───▼────┐   ┌──────▼──────┐  ┌─────▼────┐    │
│Classify│   │   Extract   │  │  Embed   │    │
│Worthy? │   │  Structure  │  │ & Store  │    │
└───┬────┘   └──────┬──────┘  └─────┬────┘    │
    │               │               │          │
    └───────────────┴───────────────┘          │
                    │                          │
                    ▼                          │
┌───────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                               LanceDB                                │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────────┐ │  │
│  │  │   Episodes    │  │     Facts     │  │        Summaries         │ │  │
│  │  │ + vectors     │  │ + vectors     │  │ + vectors                │ │  │
│  │  └───────────────┘  └───────────────┘  └──────────────────────────┘ │  │
│  │                                                                       │  │
│  │  Provenance fields are stored directly with records                   │  │
│  │  (e.g., source_episode_ids) for traceability.                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        CONSOLIDATION PROCESS                               │
│                     (Periodic / On-demand)                                 │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│   │ Group Episodes  │───▶│    Generate     │───▶│  Extract/Update     │   │
│   │   by Topic      │    │   Summaries     │    │      Facts          │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. Why LanceDB (unified) instead of split SQLite + FAISS?

**LanceDB advantages for this use case:**

- Single local backend for both metadata and vectors
- Atomic record + embedding writes reduce consistency edge cases
- SQL-like filtering plus vector search in one query path
- No server process; still local-first and offline-capable
- Simpler maintenance surface than coordinating two stores

**Trade-off:** For multi-tenant, high-concurrency deployments, Postgres/pgvector may still be preferable due to stronger transactional semantics and ecosystem tooling.

### 2. Local LLM Choice: Qwen 2.5 7B

For local-first operation, we recommend **Qwen 2.5 7B Instruct** via Ollama.

**Why Qwen over other local models?**

| Criterion                 | Qwen 2.5 7B | Llama 3.2 | Mistral 7B | Phi-3 |
| ------------------------- | ----------- | --------- | ---------- | ----- |
| JSON output reliability   | ★★★★★       | ★★★★☆     | ★★★☆☆      | ★★★☆☆ |
| Instruction following     | ★★★★★       | ★★★★☆     | ★★★★☆      | ★★★☆☆ |
| Multilingual support      | ★★★★★       | ★★★☆☆     | ★★★☆☆      | ★★☆☆☆ |
| Memory extraction quality | ★★★★☆       | ★★★★☆     | ★★★☆☆      | ★★★☆☆ |
| Speed (tokens/sec)        | ★★★★☆       | ★★★★★     | ★★★★★      | ★★★★★ |

**Key advantages for memory systems:**

1. **Structured output compliance**: Memory extraction requires reliable JSON parsing. Qwen consistently produces valid JSON even for complex schemas, while other models often add commentary or malform brackets.

2. **Multilingual handling**: Personal memory systems encounter names, places, and phrases in multiple languages. Qwen handles CJK characters, diacritics, and code-switching gracefully.

3. **Context understanding**: Memory classification requires understanding nuance ("I'm tired" vs "I've been tired for weeks"). Qwen captures these distinctions better than smaller models.

4. **Balanced performance**: At 7B parameters, Qwen runs efficiently on consumer hardware while maintaining quality. Quantized versions (Q4_K_M) work well on 8GB VRAM.

**Configuration for determinism:**

```python
temperature = 0.2  # Low for consistent extraction
top_p = 0.9        # Slight diversity for natural summaries
repeat_penalty = 1.1  # Prevent repetitive outputs
```

### 3. Local Embedding Choice: BGE-M3 via FastEmbed (ONNX)

For semantic search and retrieval, we use **BAAI/bge-m3** as the default embedding model.

**Why BGE-M3 over other embedding models?**

| Criterion        | BGE-M3 | all-MiniLM-L6-v2 | OpenAI ada-002 | Cohere |
| ---------------- | ------ | ---------------- | -------------- | ------ |
| Semantic quality | ★★★★★  | ★★★☆☆            | ★★★★★          | ★★★★☆  |
| Multilingual     | ★★★★★  | ★★☆☆☆            | ★★★★☆          | ★★★★☆  |
| Local-first      | ★★★★★  | ★★★★★            | ★☆☆☆☆          | ★☆☆☆☆  |
| Speed (CPU)      | ★★★☆☆  | ★★★★★            | ★★★★☆          | ★★★★☆  |
| Model size       | 1GB    | 80MB             | N/A            | N/A    |

**Key advantages for memory systems:**

1. **Dense retrieval optimized**: BGE-M3 is specifically trained for semantic search, producing embeddings that capture meaning well for retrieval tasks.

2. **Multilingual excellence**: Memory systems contain mixed languages (names, phrases, locations). BGE-M3 handles 100+ languages natively without quality degradation.

3. **No API dependency**: Runs entirely locally via FastEmbed (ONNX). No rate limits, no costs, works offline.

4. **Balanced dimensions**: 1024-dimensional embeddings provide good quality/storage trade-off (vs 384 for MiniLM or 1536+ for OpenAI).

**How retrieval uses normalized embeddings:**

```python
# All embeddings are L2-normalized at creation time
embedding = model.encode(text, normalize_embeddings=True)

# For normalized vectors, inner product is equivalent to cosine similarity:
# cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b))
# When norm(a) = norm(b) = 1: cosine_sim(a, b) = dot(a, b)
```

**Known limitations:**

1. **Model download**: First run downloads ~1GB model. Subsequent runs use cached model.
2. **Memory usage**: Model requires ~2GB RAM when loaded.
3. **CPU speed**: Throughput depends on model/runtime and hardware; very large batch ingestion can still be CPU-bound.

**Alternative models:**

```bash
# Faster, smaller, lower quality
export EMBEDDING_MODEL=all-MiniLM-L6-v2  # 384 dims, 80MB

# Good balance of quality and speed
export EMBEDDING_MODEL=all-mpnet-base-v2  # 768 dims, 420MB

# Via Ollama (if Ollama is running)
export EMBEDDING_PROVIDER=ollama
export OLLAMA_EMBED_MODEL=nomic-embed-text  # 768 dims
```

### 4. Initialization Simplification After Migration

The previous macOS-specific FAISS + SentenceTransformers initialization ordering constraints are no longer part of the architecture.

Current bootstrap behavior (`src/bootstrap.py`) is intentionally simple:

1. Create unified storage (`LanceStore`)
2. Create embedding provider (FastEmbed/OpenAI/Ollama/mock)
3. Create LLM provider
4. Return pipeline components

This reduces native-library interaction complexity and removes the need for tokenizers-parallelism bootstrap workarounds.

**Observability: The Doctor Command**

To verify that bootstrap and provider selection are working correctly, use:

```bash
python cli.py doctor
```

This diagnostic command reports:

- Bootstrap initialization status
- Active LLM and embedding providers with model details
- Unified store configuration and dimension consistency
- Evaluation readiness warnings
- Copy-pasteable fix suggestions for misconfigurations

**Dry-Run Mode**: Use `python cli.py doctor --dry` for a safe config-only inspection that does NOT initialize models or storage backends. This allows diagnostics to run even when dependencies are missing or misconfigured, making it ideal for CI/CD pipelines, code reviews, and debugging environment issues.

**Suggested Fixes**: The doctor command now generates actionable shell commands to fix detected issues. This improves observability by making the path from "something is wrong" to "here's how to fix it" explicit and copy-pasteable.

The doctor command is designed for:

- **Debugging**: Understanding why metrics are zero or skipped
- **Code reviews**: Verifying configuration before commits
- **Onboarding**: Learning what providers are active
- **CI/CD**: Safe pre-flight checks without heavy initialization

### 5. Memory Curation Strategy

The system implements a strict "memory worthiness" gate:

```
Raw Input → Heuristic Filter → LLM Classification → Storage
              (fast, no API)     (accurate, slower)
```

**Not everything is stored.** We aggressively filter:

- Generic greetings and acknowledgments
- Immediate task requests with no lasting value
- Hypothetical questions not about the user
- Temporary states ("I'm tired")

This prevents memory bloat and keeps retrieval quality high.

### 6. LLM Output Normalization

**Why LLM output is treated as untrusted input:**

Even the best LLMs produce inconsistent JSON output. Common issues include:

- Returning `null` instead of empty arrays `[]`
- Returning `null` instead of empty strings `""`
- Returning strings where numbers are expected (e.g., `"0.8"` instead of `0.8`)
- Returning `"true"/"false"` strings instead of boolean values
- Omitting required fields entirely

**The defense-in-depth strategy:**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Prompt Hardening│ ──▶ │   Sanitization  │ ──▶ │  Pydantic Model │
│ (Prevention)    │     │   Layer         │     │  (Validation)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Layer 1: Prompt Hardening (Prevention)**

All prompt templates include explicit instructions:

- "Never return null for any field"
- "Use empty arrays [] instead of null"
- "All fields must be present"
- Example JSON responses showing proper empty values

This reduces (but doesn't eliminate) malformed responses.

**Layer 2: Sanitization Layer**

The `src/utils/llm_sanitize.py` module provides type-safe extraction:

```python
from src.utils import as_list, as_str, as_float, as_bool, as_dict

# Extract with guaranteed types
topics = as_list(result.get("topics"))       # [] if null/wrong type
content = as_str(result.get("content"))      # "" if null/wrong type
importance = as_float(result.get("importance"), default=0.5)
is_worthy = as_bool(result.get("is_worthy"), default=False)
```

All modules that parse LLM output use these helpers:

- `src/ingestion/extractor.py`
- `src/ingestion/classifier.py`
- `src/consolidation/summarizer.py`
- `src/consolidation/fact_extractor.py`

**Layer 3: Pydantic Validation**

As a final safeguard, Pydantic models validate all fields. With sanitization,
this layer should never fail, but it catches any missed edge cases.

**Known limitations:**

1. **Semantic correctness is not validated**: Sanitization ensures type safety, not that the content makes sense.

2. **Nested structures need per-field handling**: `as_list()` ensures a list, but list items may still need sanitization.

3. **Performance overhead**: Multiple `isinstance` checks add minimal latency but are necessary for robustness.

**Testing sanitization:**

Comprehensive tests in `tests/test_llm_sanitization.py` verify:

- Null handling for all types
- Type coercion (string "true" → bool True)
- Default value application
- Pydantic model integration

### 7. Episodic vs Semantic Memory

The system distinguishes between:

| Type    | What it is              | How it's stored          | Example                                     |
| ------- | ----------------------- | ------------------------ | ------------------------------------------- |
| Episode | Timestamped event       | Raw, with context        | "On Jan 5, I started learning Korean"       |
| Fact    | Stable knowledge        | Distilled, no timestamp  | "User is learning Korean"                   |
| Summary | Narrative consolidation | Aggregated from episodes | "User's Korean journey began in January..." |

**Why this matters:** Different query types need different retrieval strategies.

- "What am I working on?" → Facts (current state)
- "What happened last week?" → Episodes (timeline)
- "Tell me about my Korean journey" → Summaries + Episodes (narrative)

### 8. Provenance Architecture

Every piece of derived knowledge (facts, summaries) links back to source episodes:

```text
facts.source_episode_ids: [episode_id, ...]
summaries.source_episode_ids: [episode_id, ...]
```

This enables:

- Explaining where a fact came from
- Tracking fact evolution over time
- Identifying contradictions
- Drilling down from summary to details

### 9. Consolidation Strategy

**When to consolidate:**

1. Topic has ≥5 unconsolidated episodes
2. Topic hasn't been consolidated in 7+ days
3. User explicitly requests it

**How conflicts are handled:**

- New supporting evidence → boost fact confidence
- Contradictory evidence → supersede old fact, create new one
- Both versions preserved (old marked `superseded_by`)

**How detail is preserved:**

- Episodes are never deleted, only marked `consolidated`
- Summaries link to all source episodes
- Hierarchical summaries (weekly → monthly) compress over time

## Key Prompts

### Memory Worthiness Classification

```
Analyze this text and determine:
1. Does it contain personal information, preferences, goals, facts, or experiences worth remembering?
2. Is it specific enough to be useful in future interactions?
3. Would forgetting this information negatively impact future assistance?
```

### Episode Extraction

```
Extract:
- Core memory content (what should be remembered)
- Memory type (episodic/fact/goal/preference/reflection)
- Topics and entities
- Importance score (0-1)
- Temporal offset if mentioned ("yesterday", "last week")
```

### Summarization

```
Create a coherent narrative summary that:
- Captures key developments
- Identifies 2-4 most significant events
- Notes patterns, progress, or changes
- Reads like a brief journal entry
```

### Retrieval Synthesis

```
Synthesize an answer that:
- References specific memories when relevant
- Acknowledges uncertainty if incomplete
- Distinguishes facts from episodes
- Prefers recent, high-confidence information
```

## Example Usage Flow

```python
# 1. Ingestion
pipeline.ingest("I started learning Korean today for my Seoul trip in March")
# → Episode created with topics=['korean', 'language_learning', 'travel']
#   memory_type=GOAL, importance=0.8

pipeline.ingest("Practiced Korean for 2 hours, learned 안녕하세요")
# → Episode created with topics=['korean', 'language_learning']
#   memory_type=EPISODIC

pipeline.ingest("I prefer visual learning over audio")
# → Episode created with topics=['learning']
#   memory_type=PREFERENCE

# 2. Consolidation (after enough episodes)
consolidation.consolidate_topic("korean")
# → Summary: "User began learning Korean in preparation for a March trip to Seoul..."
# → Fact: "User is learning Korean" (category=knowledge)
# → Fact: "User has trip to Seoul planned for March" (category=goal)
# → Fact: "User prefers visual learning" (category=preference)

# 3. Semantic Query
engine.query("What languages am I learning?")
# → Answer: "Based on your memories, you are currently learning Korean..."
# → Supporting: [Episode about starting Korean, Fact about learning Korean]

# 4. Narrative Recall
engine.recall_narrative("korean")
# → Narrative: "Your Korean learning journey began when you decided to learn
#              for an upcoming trip to Seoul. You've been practicing regularly..."
# → Timeline: [Jan 5: Started, Jan 6: First practice, Jan 7: Learned greetings]
```

## Evaluation Framework

### Metrics Selection Rationale

The evaluation module implements three core metrics, each targeting a different aspect of memory system quality:

#### 1. Retrieval Precision@K

**What it measures:** Given a query and expected relevant episodes, what fraction of the top-K results are actually relevant?

**Why this metric:**

- Directly measures retrieval quality from user's perspective
- K=5 is typical for memory recall (users don't want to scroll through 50 results)
- Precision is more important than recall for memory systems (better to show fewer, relevant memories than flood with noise)

**Formula:**

```
Precision@K = |relevant ∩ top-K| / K
```

**Interpretation:**

- 0.8+ = Excellent (4/5 top results relevant)
- 0.6-0.8 = Good (most results relevant)
- 0.4-0.6 = Fair (needs improvement)
- <0.4 = Poor (significant retrieval issues)

#### 2. Fact Conflict Rate

**What it measures:** What percentage of extracted facts conflict with other facts in the system?

**Why this metric:**

- Memory systems must maintain consistency
- Contradictory facts confuse users and degrade trust
- Detects issues in fact extraction and update logic

**Conflict definition:**

- Same entity + attribute (e.g., "user:location")
- Different values (e.g., "NYC" vs "Boston")
- Excludes supersession (intentional updates)

**Interpretation:**

- <10% = Good (minimal conflicts)
- 10-20% = Acceptable (some noise)
- > 20% = Problem (extraction or update issues)

#### 3. Consolidation Compression Ratio

**What it measures:** How efficiently does consolidation compress episode content into summaries?

**Why this metric:**

- Summaries should distill, not just concatenate
- Good compression indicates meaningful abstraction
- Too low = summaries miss important details
- Too high = summaries are just copies

**Formula:**

```
Compression Ratio = summary_tokens / source_tokens
```

**Interpretation:**

- 0.1-0.3 = Excellent (70-90% compression)
- 0.3-0.5 = Good (50-70% compression)
- 0.5-0.7 = Fair (some compression)
- > 0.7 = Poor (barely compressing)

### Evaluation Limitations

1. **Ground truth is synthetic**: The diary scenario uses predefined expected topics, not real user feedback on relevance.

2. **No semantic similarity in relevance**: A retrieved episode about "Korean" is marked relevant if expected topics include "korean", but doesn't account for semantic relatedness.

3. **Conflict detection is heuristic**: Uses pattern matching for entity-attribute extraction. Misses complex conflicts requiring reasoning.

4. **Token counting is approximate**: Uses word-based estimation (~1.3 tokens/word) rather than actual tokenizer.

5. **Single scenario**: Only the diary scenario is implemented. Real-world usage patterns vary significantly.

### Extending Evaluation

**Add human-in-the-loop evaluation:**

```python
# Present retrieved memories to user, collect relevance ratings
def collect_human_ratings(query, retrieved_episodes):
    ratings = []
    for ep in retrieved_episodes:
        rating = prompt_user(f"Is this relevant to '{query}'?", ep)
        ratings.append(rating)
    return ratings
```

**Add semantic relevance scoring:**

```python
# Use embedding similarity for soft relevance
def semantic_relevance(query_embedding, episode_embedding, threshold=0.7):
    similarity = cosine_similarity(query_embedding, episode_embedding)
    return similarity >= threshold
```

**Add longitudinal evaluation:**

```python
# Track metrics over time as system accumulates memories
def track_metrics_over_time(days=30):
    metrics_timeline = []
    for day in range(days):
        simulate_day_of_activity()
        metrics = run_evaluation()
        metrics_timeline.append((day, metrics))
    return metrics_timeline
```

**Add adversarial scenarios:**

```python
# Test robustness to edge cases
adversarial_scenarios = [
    "contradictory_facts",      # User says opposite things
    "temporal_confusion",        # "yesterday" vs "last week"
    "topic_drift",              # Gradual topic changes
    "information_overload",     # 100+ episodes/day
]
```

## System Evaluation

### What This System Does Well

1. **Structured memory over vector blobs**

   - Every memory has type, time, topics, importance
   - Not just "throw everything in a vector DB"

2. **Time-aware retrieval**

   - Can answer "What was I working on last month?"
   - Preserves narrative coherence in recall

3. **Provenance tracking**

   - Can explain where any fact came from
   - Can track how knowledge evolved

4. **Memory curation**

   - Doesn't store everything
   - Prevents noise from degrading retrieval quality

5. **Clean abstraction boundaries**
   - Embedding provider is pluggable
   - LLM provider is pluggable
   - Storage layer is isolated

### What It Cannot Do Yet

1. **No real-time inference**

   - Consolidation is batch, not streaming
   - No continuous background processing

2. **Limited conflict resolution**

   - Contradictions are detected but not intelligently resolved
   - No confidence decay over time without activity

3. **No multi-modal memory**

   - Text only; no images, audio, or structured data
   - Would need separate embedding models

4. **No personalization of importance**

   - Importance is LLM-assigned, not learned from user behavior
   - Doesn't adapt to individual usage patterns

5. **No forgetting/decay mechanism**

   - Old memories persist forever
   - Could implement time-based confidence decay

6. **Single-user only**
   - No multi-tenancy or access control
   - Would need significant changes for shared use

### Extending to a Full Assistant Memory System

1. **Add real-time consolidation**

   ```python
   # Background worker that monitors for consolidation triggers
   async def consolidation_worker():
       while True:
           topics = db.get_topics_needing_consolidation()
           for topic in topics:
               await consolidate_topic(topic)
           await asyncio.sleep(3600)  # Check hourly
   ```

2. **Add confidence decay**

   ```python
   # Facts not reinforced decay over time
   def decay_old_facts():
       old_facts = db.get_facts_not_seen_in(days=90)
       for fact in old_facts:
           fact.confidence *= 0.95
           db.save_fact(fact)
   ```

3. **Add multi-modal support**

   ```python
   # Use CLIP for images, Whisper for audio
   class MultiModalEpisode(Episode):
       images: list[str]  # S3 URLs
       audio_transcript: Optional[str]
   ```

4. **Add learned importance**

   ```python
   # Track which memories are retrieved/useful
   def on_memory_used(episode_id):
       episode = db.get_episode(episode_id)
       episode.importance = min(1.0, episode.importance + 0.1)
       db.save_episode(episode)
   ```

5. **Add forgetting**
   ```python
   # Archive very old, low-importance, unretrieved memories
   def archive_forgotten():
       candidates = db.get_episodes(
           importance_lt=0.3,
           last_retrieved_before=days_ago(180),
           created_before=days_ago(365)
       )
       for ep in candidates:
           ep.is_active = False
           db.save_episode(ep)
    ```

## Migration Note (Breaking Changes)

### Storage Backend

- **From:** SQLite (metadata) + FAISS (vectors)
- **To:** LanceDB for unified metadata + vector storage
- **Why:** single-store consistency, atomic writes, simpler retrieval/filter paths, lower operational complexity.

### Local Embedding Runtime

- **From:** SentenceTransformers
- **To:** FastEmbed (ONNX)
- **Why:** smaller runtime/dependency surface and simpler local configuration.

### JSON Parsing Flow

- **From:** repeated `complete()` + `json.loads(...)` at many callsites
- **To:** centralized `complete_json()` with retryable extraction
- **Why:** better resilience to malformed model output and less duplicated parsing code.

### Prompt Rule Maintenance

- **From:** copy-pasted JSON-output rule blocks in multiple prompts
- **To:** shared `JSON_OUTPUT_RULES` composed into templates
- **Why:** consistent constraints and easier maintenance.

## Conclusion

This episodic memory pipeline provides a solid foundation for personal AI memory:

- **Correctness**: Proper separation of episodic/semantic memory, provenance tracking
- **Clarity**: Clean module boundaries, explicit data flow
- **Extensibility**: Pluggable providers, clear extension points

It's not a toy demo - it handles the hard problems (curation, consolidation, narrative recall) while remaining simple enough to understand and modify.

The main limitations are operational (single-user, batch consolidation) rather than architectural. The design supports extension to a production system with additional infrastructure.
