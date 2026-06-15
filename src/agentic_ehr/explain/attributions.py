"""Per-patient feature attributions.

Uses SHAP TreeExplainer when available (faithful, signed, per-patient). Falls
back to a dependency-light approximation: global importance weighted by how far
the patient's feature value sits from the training-population median, signed by
the concept's direction-of-concern. The fallback is clearly labelled as
approximate so downstream language can hedge appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Contribution:
    feature: str
    value: float
    signed_impact: float   # >0 pushes risk up, <0 pushes risk down
    method: str            # "shap" | "approx"


class Attributor:
    def __init__(self, model, background: pd.DataFrame, method: str = "auto"):
        self.model = model
        self.feature_names = model.feature_names
        self._medians = background.median(numeric_only=True)
        self._importance = model.feature_importance()
        self._explainer = None
        self.method = self._resolve_method(method, background)

    def _resolve_method(self, method: str, background: pd.DataFrame) -> str:
        if method in ("shap", "auto"):
            try:
                import shap  # noqa: F401

                self._init_shap(background)
                return "shap"
            except Exception as exc:  # ImportError or explainer failure
                if method == "shap":
                    logger.warning("SHAP requested but unavailable (%s); using approx.", exc)
                else:
                    logger.info("SHAP not available; using approx attributions.")
        return "approx"

    def _init_shap(self, background: pd.DataFrame) -> None:
        import shap

        # TreeExplainer needs the underlying sklearn/booster model.
        underlying = getattr(self.model, "sklearn_model", None)
        if underlying is None:
            raise RuntimeError("model exposes no sklearn_model for SHAP")
        self._explainer = shap.TreeExplainer(underlying)

    def explain(self, x_row: pd.DataFrame, top_k: int = 5) -> list[Contribution]:
        """Return the top_k contributors (by absolute signed impact)."""
        if self.method == "shap":
            contribs = self._explain_shap(x_row)
        else:
            contribs = self._explain_approx(x_row)
        contribs.sort(key=lambda c: abs(c.signed_impact), reverse=True)
        return contribs[:top_k]

    def _explain_shap(self, x_row: pd.DataFrame) -> list[Contribution]:
        values = self._explainer.shap_values(x_row.values)
        values = np.asarray(values)
        if values.ndim == 3:          # (n, features, classes)
            values = values[0, :, -1]
        else:                          # (n, features)
            values = values[0]
        row = x_row.iloc[0]
        return [
            Contribution(feature=f, value=_safe_float(row[f]), signed_impact=float(v), method="shap")
            for f, v in zip(self.feature_names, values)
        ]

    def _explain_approx(self, x_row: pd.DataFrame) -> list[Contribution]:
        from .concept_map import ConceptMap

        cm = ConceptMap()
        row = x_row.iloc[0]
        out: list[Contribution] = []
        for f in self.feature_names:
            val = _safe_float(row[f])
            med = _safe_float(self._medians.get(f, 0.0))
            imp = self._importance.get(f, 0.0)
            deviation = val - med
            direction = 1.0 if cm.resolve(f).higher_is_concerning else -1.0
            signed = direction * imp * np.tanh(deviation)
            out.append(Contribution(feature=f, value=val, signed_impact=float(signed), method="approx"))
        return out


def _safe_float(x) -> float:
    try:
        v = float(x)
        return 0.0 if np.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0
