"""
Services layer for the episodic memory pipeline.

This module contains business logic that is independent of the CLI.
Services return plain dataclasses/dicts and do not import Rich/Typer.

Usage:
    from src.services import IngestionService, RetrievalService

    service = IngestionService(components)
    result = service.ingest_text("Some memory")
"""

from .diagnostics import DiagnosticsService
from .evaluation import EvaluationService
from .ingestion import IngestionService
from .retrieval import RetrievalService

__all__ = [
    "IngestionService",
    "RetrievalService",
    "EvaluationService",
    "DiagnosticsService",
]
