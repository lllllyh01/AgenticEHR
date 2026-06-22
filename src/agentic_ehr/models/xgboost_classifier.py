"""XGBoost classification baseline implementing the :class:`BaseModel` interface.

Includes optional probability calibration (isotonic/sigmoid) fitted on a
held-out split, and a simple per-patient uncertainty estimate derived from the
spread across the gradient-boosted trees (epistemic-ish) plus distance to the
decision boundary.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ..logging_utils import get_logger
from .base import BaseModel, ModelOutput

logger = get_logger(__name__)


class XGBoostClassifierModel(BaseModel):
    name = "xgboost"

    def __init__(self, params: dict | None = None, calibrate: bool = True,
                 calibration_method: str = "isotonic"):
        self.params = params or {}
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.model_: XGBClassifier | None = None
        self._feature_names: list[str] = []
        self._calibrator = None

    # ----- training ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None) -> "XGBoostClassifierModel":
        self._feature_names = list(X.columns)
        self.model_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            **self.params,
        )
        self.model_.fit(X.values, y)
        logger.info("Trained XGBoost on %d rows, %d features", *X.shape)

        if self.calibrate and X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            self._fit_calibrator(X_val, y_val)
        return self

    def _fit_calibrator(self, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        raw = self._raw_proba(X_val)
        if self.calibration_method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, y_val)
        else:  # platt / sigmoid
            cal = LogisticRegression()
            cal.fit(raw.reshape(-1, 1), y_val)
        self._calibrator = cal
        logger.info("Fitted %s calibrator on %d held-out rows", self.calibration_method, len(y_val))

    # ----- inference ---------------------------------------------------------
    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._as_array(X))[:, 1]

    def _apply_calibration(self, raw: np.ndarray) -> np.ndarray:
        if self._calibrator is None:
            return raw
        if isinstance(self._calibrator, IsotonicRegression):
            return self._calibrator.predict(raw)
        return self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply_calibration(self._raw_proba(X))

    def predict_output(self, X: pd.DataFrame) -> list[ModelOutput]:
        raw = self._raw_proba(X)
        cal = self._apply_calibration(raw)
        unc = self._uncertainty(X, cal)
        return [
            ModelOutput(probability=float(c), raw_probability=float(r), uncertainty=float(u))
            for c, r, u in zip(cal, raw, unc)
        ]

    def _uncertainty(self, X: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
        """Heuristic uncertainty in [0, 1].

        Combines (a) closeness to the 0.5 decision boundary and (b) disagreement
        across boosting stages (variance of staged predictions). Both are cheap
        and need no extra model. Documented as heuristic, not a guarantee.
        """
        arr = self._as_array(X)
        boundary = 1.0 - np.abs(prob - 0.5) * 2.0  # 1 at p=0.5, 0 at p in {0,1}

        try:
            booster = self.model_.get_booster()
            leaf = booster.predict(_to_dmatrix(arr), pred_leaf=True)
            spread = leaf.std(axis=1)
            spread = spread / (spread.max() + 1e-9)
        except Exception:  # pragma: no cover - defensive
            spread = np.zeros_like(prob)

        unc = 0.7 * boundary + 0.3 * spread
        return np.clip(unc, 0.0, 1.0)

    # ----- explainability hook ----------------------------------------------
    def feature_importance(self) -> dict[str, float]:
        booster_scores = self.model_.feature_importances_
        return {name: float(score) for name, score in zip(self._feature_names, booster_scores)}

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def sklearn_model(self) -> XGBClassifier:
        return self.model_

    # ----- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "params": self.params,
                "calibrate": self.calibrate,
                "calibration_method": self.calibration_method,
                "feature_names": self._feature_names,
                "model": self.model_,
                "calibrator": self._calibrator,
            },
            path,
        )
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifierModel":
        blob = joblib.load(path)
        obj = cls(
            params=blob["params"],
            calibrate=blob["calibrate"],
            calibration_method=blob["calibration_method"],
        )
        obj.model_ = blob["model"]
        obj._feature_names = blob["feature_names"]
        obj._calibrator = blob["calibrator"]
        logger.info("Loaded XGBoost model from %s", path)
        return obj

    # ----- helpers -----------------------------------------------------------
    @staticmethod
    def _as_array(X) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)


def _to_dmatrix(arr):
    import xgboost as xgb

    return xgb.DMatrix(arr)
