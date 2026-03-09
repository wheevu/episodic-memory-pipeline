# Episodic Memory Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Local-first memory for AI agents.**

Separates what an agent **experiences** (episodic memory) from what it **knows** (semantic memory), then consolidates episodes into durable facts and summaries for better recall.

## Stack

- **LanceDB** for vector + metadata storage
- **FastEmbed** for local embeddings
- **Qwen 2.5** or **OpenAI** for extraction and reasoning

## Features

- episodic -> semantic memory consolidation
- local-first, lightweight architecture
- traceable facts linked to source episodes
- narrative recall beyond nearest-neighbor retrieval

## Example

Input episodes:
- "I started learning Korean today."
- "I want to be conversational before my trip to Seoul in March."

After consolidation:
- Fact: user is learning Korean
- Goal: become conversational before March trip to Seoul

Query:
> What am I learning, and why?

Answer:
> You're learning Korean, and your near-term goal is to become conversational before your March trip to Seoul.

## Quick Start

```bash
git clone https://github.com/wheevu/episodic-memory-pipeline
cd episodic-memory-pipeline
pip install -e .
cp env.example .env
make demo
```
### Mock/demo mode:
```bash
make demo-mock
```

## CLI
```bash
episodic-memory ingest "I started learning Korean today"
episodic-memory query "What am I learning?"
episodic-memory consolidate --all
episodic-memory stats
```

## License
MIT
