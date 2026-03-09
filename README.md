# Episodic Memory Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Giving local AI agents human-like memory by separating what they experience (episodic) from what they know (semantic).**

This system captures timestamped experiences, consolidates them into stable facts and summaries, and retrieves them with narrative coherence. The current stack uses **LanceDB** for unified metadata + vector storage, **FastEmbed (ONNX)** for local embeddings, and **Qwen 2.5** (or OpenAI) for reasoning/extraction.

## How It Works

### 1) Experience (Episodic Memory)
The agent captures raw, timestamped events as they happen.

> Tuesday, 3:00 PM: "I just started learning Korean. I want to be conversational before my flight to Seoul in March."

### 2) Consolidation ("Sleep" Phase)
Periodically (e.g., daily), the system compresses recent episodes.

> It extracts durable facts, detects updates/contradictions, and writes topic summaries.

### 3) Recall (Semantic + Narrative)
Later queries retrieve facts, episodes, and summaries and synthesize a coherent answer.

## Quick Start

```bash
git clone https://github.com/wheevu/episodic-memory-pipeline
cd episodic-memory-pipeline
pip install -e .
cp env.example .env

# Generate local artifacts deterministically
make demo
```

For a fast dependency-light run (mock providers), use:

```bash
make demo-mock
```

## CLI

```bash
episodic-memory doctor --dry
episodic-memory ingest "I started learning Korean today"
episodic-memory query "What am I learning?"
episodic-memory recall "korean" --topic
episodic-memory consolidate --all
episodic-memory stats
```

Legacy entrypoint still works:

```bash
python cli.py doctor --dry
python cli.py query "What am I learning?"
```

## Evaluation (Versioned Runs)

Runs are stored under `runs/eval/<run_id>/eval_run.json` and include config snapshot, metrics, warnings, and commit metadata when available.

```bash
episodic-memory eval-run --scenario diary
episodic-memory eval-list
episodic-memory eval-compare <runA> <runB>
```

## Project Structure

- `src/models/`: core data models (`Episode`, `Fact`, `Summary`)
- `src/services/`: service layer (ingestion/retrieval/evaluation/diagnostics)
- `src/storage/`: unified LanceDB storage abstraction (`LanceStore`)
- `scripts/`: bootstrap utilities
- `demo_data/`: synthetic fixtures for demo/tests
- `runs/eval/`: evaluation outputs

## Key Features

- **Local-first:** no server required for core operation
- **Unified storage:** single LanceDB backend for vectors + metadata
- **Traceability:** facts/summaries preserve source episode lineage
- **Resilience:** centralized sanitization and retry-aware JSON extraction
- **Narrative recall:** retrieval can answer as timeline/journey, not only nearest vectors

## Configuration

Copy `env.example` to `.env` (or export vars directly):

```bash
# Embeddings
EMBEDDING_PROVIDER=local            # local|openai|ollama|mock
EMBEDDING_MODEL=BAAI/bge-m3         # local/openai model name
OLLAMA_EMBED_MODEL=nomic-embed-text # when EMBEDDING_PROVIDER=ollama

# LLM
LLM_PROVIDER=ollama                 # openai|ollama
LLM_MODEL=gpt-4o-mini               # when LLM_PROVIDER=openai
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.2

# API keys
OPENAI_API_KEY=sk-your-key-here

# Storage
LANCE_DB_PATH=./data/lancedb
```

## Programmatic Usage

```python
from episodic_memory import MemorySystem

mem = MemorySystem()  # reads .env/config
mem.remember("I need to buy groceries for Friday dinner")

result = mem.recall("What am I planning this week?")
print(result.answer)
```

For lower-level wiring:

```python
from episodic_memory.bootstrap import get_components

components = get_components()
# components.lance_store, components.embedding_provider, components.llm
```

Backward-compatible imports from `src` still work, but `episodic_memory` is the preferred
public package namespace.

## Development

```bash
make install-dev
make test
make test-slow
make lint
make format
```

## License

MIT

## Migration Note (What Changed, From What, and Why)

- `SQLite + FAISS` -> `LanceDB` (unified store)
  - **Changed from:** split metadata DB + separate vector index.
  - **Changed to:** single `LanceStore` API with atomic save/update/search.
  - **Why:** fewer consistency edge cases, simpler query path, easier maintenance.

- `SentenceTransformers` -> `FastEmbed (ONNX)` for local embeddings
  - **Changed from:** device-specific SentenceTransformers model loading.
  - **Changed to:** FastEmbed `TextEmbedding` with simpler local runtime.
  - **Why:** lighter dependency/runtime surface and simpler configuration.

- `complete() + json.loads(...)` callsites -> `complete_json()` with retries
  - **Changed from:** repeated ad-hoc JSON parsing in ingestion/consolidation/retrieval.
  - **Changed to:** centralized extraction with retry on malformed responses.
  - **Why:** improved robustness and reduced duplicated parsing logic.

- Prompt JSON constraints duplicated per template -> shared `JSON_OUTPUT_RULES`
  - **Changed from:** copy-pasted rule blocks in each prompt template.
  - **Changed to:** template composition with one shared rules constant.
  - **Why:** consistency and easier maintenance.
