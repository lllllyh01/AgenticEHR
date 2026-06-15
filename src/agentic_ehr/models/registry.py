"""Model registry so models can be selected by name from config.

Register additional models (MOTOR-T, a foundation-model adapter, ...) here or
via ``register_model`` at import time, then set ``model.name`` in the config.
"""
from __future__ import annotations

from typing import Callable

from .base import RiskModel
from .xgboost_model import XGBoostRiskModel

_REGISTRY: dict[str, Callable[..., RiskModel]] = {}


def register_model(name: str, factory: Callable[..., RiskModel]) -> None:
    _REGISTRY[name] = factory


def get_model(name: str, **kwargs) -> RiskModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _REGISTRY[name](**kwargs)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_from_config(cfg) -> RiskModel:
    name = cfg.get("model.name", "xgboost")
    if name == "xgboost":
        return get_model(
            "xgboost",
            params=cfg.get("model.params", {}),
            calibrate=cfg.get("model.calibrate", True),
            calibration_method=cfg.get("model.calibration_method", "isotonic"),
        )
    return get_model(name)


register_model("xgboost", XGBoostRiskModel)
