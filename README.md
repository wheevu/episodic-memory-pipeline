# Episodic Memory Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Cognitive-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Maintained-success?style=flat-square)

**A local-first cognitive architecture for AI agents.**

This is not just a vector database wrapper. It is a system that mimics human memory consolidation by separating **Episodic Memory** (raw, timestamped events) from **Semantic Memory** (consolidated, stable facts). It features defense-in-depth LLM sanitization, provenance tracking, and is optimized for multilingual (CJK) contexts using **Qwen 2.5** and **BGE-M3**.

## Quick Start (Reproducible Demo)

```bash
git clone <repo-url>
cd episodic-memory-pipeline
pip install -e .

# Generate local artifacts deterministically (no committed binaries)
make demo
```

## CLI

After installation, the console script is available:

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

Evaluation runs are stored under `runs/eval/<run_id>/eval_run.json` and include:

- git commit hash (if available)
- config snapshot (provider/model, k, scenario)
- metrics + warnings

```bash
episodic-memory eval-run --scenario diary
episodic-memory eval-list
episodic-memory eval-compare <runA> <runB>
```

## Design Philosophy

1. **Episodic memory ≠ vector blobs**: Each memory is a structured event with context, time, and meaning.
2. **Time and provenance matter**: Every fact and summary links back to its source episodes. Hallucination prevention starts with lineage.
3. **Memory must be curated, not accumulated**: Not everything is worth remembering. We filter aggressively via a “Memory Worthiness” gate.
4. **Retrieval should feel like recalling a journey**: Narrative coherence over raw similarity scores.

## Storage Choice: SQLite + FAISS

**Why SQLite over Postgres?**

- Local-first, no server dependencies
- Single-file portability (backup = copy file)
- JSON1 extension for flexible metadata
- Zero configuration required

**Why FAISS for vectors?**

- Mature, fast, local-only C++ library
- Supports multiple index types for scaling
- Works well alongside SQLite for hybrid retrieval

## Core Concepts

### Episode (Episodic Memory)

A timestamped event capturing what happened, when, and in what context.

> _“On Tuesday at 3pm, I told my assistant I'm learning Korean for a trip to Seoul in March.”_

### Fact (Semantic Memory)

A distilled, stable piece of knowledge extracted from episodes.

> _“User is learning Korean. User has a trip to Seoul planned for March 2024.”_

### Summary (Consolidated Narrative)

A topic-level summary that weaves together multiple episodes into a coherent narrative.

> _“User's Korean language learning journey: Started in January 2024 motivated by upcoming Seoul trip...”_

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Embeddings (default: local)
EMBEDDING_PROVIDER=local            # local|openai|ollama|mock
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu                # cpu|cuda|mps

# LLM
LLM_PROVIDER=ollama                 # openai|ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.2
```

## Local-First Setup with Qwen (Recommended)

```bash
# Install Ollama
brew install ollama  # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux

ollama pull qwen2.5:7b-instruct
ollama serve

export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:7b-instruct
export EMBEDDING_PROVIDER=local
```

## Demo Data Policy

The `demo_data/` directory contains **synthetic data only** for demonstration and testing.

- ✅ Fictional diary entries and memories
- ✅ Example evaluation queries
- ❌ Never commit real user data
- ❌ No sensitive information (API keys, PII)

See `demo_data/README.md` for details.

## Development

```bash
make test
make test-slow
make lint
make format

make demo
make demo-clean
make demo-mock
```

## Project Structure (High Level)

```
src/cli/        # CLI commands + rendering (Rich/Click)
src/services/   # Business logic (no Rich/Click; returns plain dataclasses/dicts)
scripts/        # Reproducible bootstrap utilities
demo_data/      # Synthetic fixtures (safe-to-commit)
runs/eval/      # Versioned eval run outputs (gitignored per-run)
data/           # Generated local artifacts (gitignored)
```

## macOS: FAISS + SentenceTransformers Note

On macOS, there's a known interaction issue between FAISS and SentenceTransformers during Python cleanup. **This is handled automatically** by the bootstrap module.

For library users, import from `src.bootstrap`:

```python
from src.bootstrap import get_components

components = get_components()
# components.database, components.embedding_provider, etc.
```

## License

MIT
