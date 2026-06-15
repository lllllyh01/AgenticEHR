"""The model interface every predictive model must implement.

The agent never imports this module — it only ever sees a ``RiskProfile``. This
ABC is the contract for *training/serving* a model. To plug in MOTOR-T or a
foundation model later, implement this interface (see README) and register it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelOutput:
    """Per-patient model output.

    ``probability``        calibrated probability of the positive outcome.
    ``raw_probability``    pre-calibration probability (for diagnostics).
    ``uncertainty``        scalar in [0, 1]; higher = less confident.
    """

    probability: float
    raw_probability: float
    uncertainty: float = 0.0
    extra: dict = field(default_factory=dict)


class RiskModel(ABC):
    """Abstract pluggable risk model."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None) -> "RiskModel":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated positive-class probabilities, shape (n,)."""

    @abstractmethod
    def predict_output(self, X: pd.DataFrame) -> list[ModelOutput]:
        """Return rich per-row :class:`ModelOutput` (proba + uncertainty)."""

    @abstractmethod
    def feature_importance(self) -> dict[str, float]:
        """Global feature importance, mapping feature name -> score."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "RiskModel":
        ...
