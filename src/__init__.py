"""
Episodic Memory Pipeline
A local-first personal memory system for AI assistants.
"""

__version__ = "0.1.0"

# Public API — the MemorySystem facade is the primary entry point for agents.
# It is imported lazily to avoid triggering heavy model/FAISS loads at
# package-import time.  Usage:
#
#   from src import MemorySystem
#   mem = MemorySystem()
#

from src.memory import MemorySystem

__all__ = ["MemorySystem", "__version__"]
