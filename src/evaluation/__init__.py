"""
Evaluation module for the episodic memory pipeline.

Provides metrics and evaluation runners for assessing memory system quality.
"""

from .metrics import (
    ConsolidationCompressionMetric,
    EvaluationMetrics,
    FactConflictRateMetric,
    RetrievalPrecisionMetric,
)
from .runner import DiaryScenario, EvaluationRunner, EvaluationScenario

__all__ = [
    "RetrievalPrecisionMetric",
    "FactConflictRateMetric",
    "ConsolidationCompressionMetric",
    "EvaluationMetrics",
    "EvaluationRunner",
    "EvaluationScenario",
    "DiaryScenario",
]
