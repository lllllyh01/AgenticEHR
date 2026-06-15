"""Evaluation: predictive metrics + pragmatic summary-quality checks."""
from .model_eval import evaluate_predictions, ModelMetrics
from .summary_eval import evaluate_summary, SummaryQuality

__all__ = ["evaluate_predictions", "ModelMetrics", "evaluate_summary", "SummaryQuality"]
