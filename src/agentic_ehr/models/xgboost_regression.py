"""XGBoost regression model for continuous targets (e.g. length of stay).

Mirrors :class:`XGBoostRiskModel` for the methods the multi-task trainer,
attributor, and service need (fit / predict / predict_output / feature_importance
/ sklearn_model / save / load), but predicts a continuous ``point_estimate`` with
no probability calibration.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..logging_utils import get_logger
from .base import ModelOutput

logger = get_logger(__name__)


class XGBoostRegressionModel:
    name = "xgboost_regression"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model_: XGBRegressor | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None) -> "XGBoostRegressionModel":
        self._feature_names = list(X.columns)
        self.model_ = XGBRegressor(objective="reg:squarederror", tree_method="hist", **self.params)
        self.model_.fit(X.values, np.asarray(y, dtype=float))
        logger.info("Trained XGBoost regressor on %d rows, %d features", *X.shape)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._as_array(X))

    def predict_output(self, X: pd.DataFrame) -> list[ModelOutput]:
        pred = self.predict(X)
        unc = self._uncertainty(X)
        return [
            ModelOutput(probability=0.0, raw_probability=0.0,
                        uncertainty=float(u), point_estimate=float(p))
            for p, u in zip(pred, unc)
        ]

    def _uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Heuristic uncertainty in [0, 1] from per-tree leaf spread."""
        arr = self._as_array(X)
        try:
            import xgboost as xgb
            leaf = self.model_.get_booster().predict(xgb.DMatrix(arr), pred_leaf=True)
            spread = leaf.std(axis=1)
            spread = spread / (spread.max() + 1e-9)
            return np.clip(spread, 0.0, 1.0)
        except Exception:  # pragma: no cover - defensive
            return np.zeros(len(arr))

    def feature_importance(self) -> dict[str, float]:
        return {n: float(s) for n, s in zip(self._feature_names, self.model_.feature_importances_)}

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def sklearn_model(self) -> XGBRegressor:
        return self.model_

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "feature_names": self._feature_names, "model": self.model_}, path)
        logger.info("Saved XGBoost regressor to %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostRegressionModel":
        blob = joblib.load(path)
        obj = cls(params=blob["params"])
        obj.model_ = blob["model"]
        obj._feature_names = blob["feature_names"]
        return obj

    @staticmethod
    def _as_array(X) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
