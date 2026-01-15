#!/usr/bin/env python3
"""
Command-line interface for the episodic memory pipeline.

This is a thin wrapper that delegates to src.cli.
For the full implementation, see src/cli/.

Usage:
    python cli.py ingest "I started learning Korean today"
    python cli.py query "What am I learning?"
    python cli.py recall "Tell me about my Korean learning journey"
    python cli.py consolidate --topic language_learning
    python cli.py stats
    python cli.py demo
    python cli.py eval --scenario diary
    python cli.py eval-run --scenario diary --save
    python cli.py eval-compare <run_a> <run_b>
    python cli.py doctor
    python cli.py doctor --dry

After installation (pip install -e .), you can also use:
    episodic-memory ingest "..."
    episodic-memory query "..."
"""

from src.cli import app

if __name__ == "__main__":
    app()
