"""Pluggable risk-model interface and the XGBoost baseline."""
from .base import RiskModel, ModelOutput
from .xgboost_model import XGBoostRiskModel
from .registry import get_model, register_model, available_models

__all__ = [
    "RiskModel",
    "ModelOutput",
    "XGBoostRiskModel",
    "get_model",
    "register_model",
    "available_models",
]
