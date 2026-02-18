# Episodic Memory Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Giving local AI agents human-like memory by separating what they experience (episodic) from what they know (semantic).**

This memory system pipeline models human cognitive processes: it captures raw experiences, and acts like a brain during sleep, storing events into stable facts and keeping a coherent narratives over time.

## How It Works

Imagine an agent helping a user plan a trip. Here is how the system processes that interaction:

### 1. Experience (Episodic Memory)
The agent captures raw, timestamped events as they happen.
> *Tuesday, 3:00 PM: "I just started learning Korean. I want to be conversational before my flight to Seoul in March."*

### 2. Consolidation (The "Sleep" Phase)
Ideally run periodically (e.g., daily), the system analyzes recent episodes to compress them.
> *System identifies a new goal (Language Learning) and a future event (Trip).*

### 3. Knowledge (Semantic Memory & Summaries)
The raw events are distilled into stable knowledge that is easy to query later.
*   **Fact:** User is learning Korean.
*   **Fact:** User has a trip to Seoul planned for March 2024.
*   **Summary:** "In January 2024, the user began an intensive Korean study regimen motivated by an upcoming trip to Seoul."

## Project Structure

*   `src/models/`: Data definitions (Episode, Fact, Summary).
*   `src/services/`: Core logic (Ingestion, Consolidation, Retrieval).
*   `src/storage/`: Database (SQLite) and Vector (FAISS) wrappers.
*   `demo_data/`: Synthetic fixtures for testing.

## Key Features

*   **Local-First:** Built on SQLite and FAISS. No server dependencies, single-file portability.
*   **Traceability:** Every fact and summary links back to the specific episodes that created it. Zero hallucinated memories.
*   **Self-Correction:** Includes "defense-in-depth" sanitization. If the LLM outputs malformed JSON, the pipeline repairs it automatically.
*   **Evaluation-Ready:** Comes with a full evaluation framework to benchmark memory recall and precision.

---

## Quick Start

You can run the full pipeline locally without an LLM (using mocks) to see the architecture in action immediately.

```bash
git clone https://github.com/wheevu/episodic-memory-pipeline
cd episodic-memory-pipeline
pip install -e .

# Run a fast demo with mock providers
make demo-mock
```

To use a real LLM (Ollama or OpenAI), see [Configuration](#configuration).

## Usage Workflow

The system is designed to be used in a continuous loop: **Remember → Consolidate → Recall**.

### 1. Remember (Ingest)
Store text from a user interaction. The system automatically classifies if it's worth remembering (filtering out "Hi" or "Okay").
```bash
episodic-memory ingest "I started learning Korean today"
```

### 2. Consolidate (Process)
Trigger the consolidation process to extract facts and summaries from recent episodes.
```bash
episodic-memory consolidate --all
```

### 3. Recall (Query)
Retrieve information using natural language. The system decides whether to fetch specific facts or recount a narrative journey.
```bash
# Ask a specific question
episodic-memory query "What am I learning?"

# Recall a narrative journey
episodic-memory recall "korean" --topic
```

---

## Configuration

The system is local-first but flexible. Copy `env.example` to `.env` to configure your backend.

### Option A: Local (Recommended)
Free and private. Requires [Ollama](https://ollama.com/).

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
```

### Option B: OpenAI
```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_PROVIDER=openai
```

## Developer Interface

To integrate this into your own Python agent:

```python
from src.memory import MemorySystem

# Automatically loads config from .env
mem = MemorySystem() 

# 1. Store an interaction
mem.remember("I need to buy groceries for the dinner party")

# 2. Retrieve context for your agent prompt
# Returns relevant facts, recent episodes, and topic summaries
context = mem.get_context("groceries")

# 3. Direct query
result = mem.recall("What do I need to do?")
print(result.answer)
```

## License

MIT
