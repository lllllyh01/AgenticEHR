"""Pluggable predictive-model interface and the XGBoost baselines."""
from .base import BaseModel, ModelOutput
from .xgboost_classifier import XGBoostClassifierModel
from .xgboost_regression import XGBoostRegressionModel
from .registry import get_model, register_model, available_models

__all__ = [
    "BaseModel",
    "ModelOutput",
    "XGBoostClassifierModel",
    "XGBoostRegressionModel",
    "get_model",
    "register_model",
    "available_models",
]
