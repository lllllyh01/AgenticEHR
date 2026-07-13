"""Multi-task inference service: patient -> HealthRiskProfile (prediction panel).

Loads the trained per-task models and, for one patient, predicts every task and
attributes each prediction, assembling the multi-label ``HealthRiskProfile`` the
health-summary agent consumes.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ...config import Config
from ...explain.attributions import Attributor
from ...explain.concept_map import Concept, ConceptMap
from ...explain.risk_profile import HealthRiskProfile, RiskProfileBuilder, TaskPrediction
from ...logging_utils import get_logger
from ..dataset import _load_event_label_records, build_snapshot
from . import concepts as C
from . import tasks as T
from .multitask import MultiTaskModel, _anchor_task, _windows_used

logger = get_logger(__name__)

_STAT_WORDS = {"mean": "average", "min": "lowest", "max": "highest",
               "last": "latest", "std": "variability in"}


def mimic_concept_map() -> dict[str, Concept]:
    """Plain-language concepts for every MIMIC feature code (for the agent)."""
    m: dict[str, Concept] = {}
    for v in C.VITAL_SERIES:
        concerning = v.code != "SPO2"   # for SpO2 a LOWER value is concerning
        for stat in C.VITAL_STATS:
            word = _STAT_WORDS[stat]
            phrase = (f"the {word} your {v.description.lower()}" if stat == "std"
                      else f"your {word} {v.description.lower()}")
            m[f"VITAL/{v.code}_{stat}"] = Concept(
                f"{v.description} ({stat})", phrase, higher_is_concerning=concerning)
    for lab in C.LAB_PANEL:
        m[f"LAB/{lab.code}"] = Concept(lab.description, f"your {lab.description.lower()} level")
    for g in C.ICD_GROUPS:
        m[f"DX/{g.code}"] = Concept(g.description, f"a history of {g.description.lower()}")
    m["UTIL/N_PRIOR_ADM"] = Concept("Prior hospital admissions",
                                    "your number of previous hospital admissions")
    return m


# Cap the attribution-median background so building the service stays fast even
# on the full ~58k cohort (a deterministic stride sample is plenty for medians).
_MAX_BACKGROUND = 4000


_SNAPSHOT_WINDOW = None   # discharge window: richest record; used for the agent snapshot


class MultiTaskInferenceService:
    def __init__(self, cfg: Config, model: MultiTaskModel, records_by_window: dict[str, list]):
        self.cfg = cfg
        self.model = model
        # window -> {patient_id: record}
        self.records = {w: {r.patient_id: r for r in recs} for w, recs in records_by_window.items()}
        self.lookback = cfg.get("data.featurize.lookback_days", 3650)
        self.top_k = cfg.get("explain.top_k_contributors", 5)
        self.concept_map = ConceptMap(mimic_concept_map())
        self.risk_tiers = cfg.get("agent.risk_tiers")
        method = cfg.get("explain.method", "auto")

        # Attribution-median background, one per window (deterministic subsample).
        backgrounds = {}
        for w, recs in records_by_window.items():
            step = max(1, len(recs) // _MAX_BACKGROUND)
            backgrounds[w] = model.featurizer_for(w).transform(recs[::step])
        self.builders: dict[str, RiskProfileBuilder] = {
            name: RiskProfileBuilder(
                Attributor(tm.model, backgrounds[tm.spec.window][tm.columns], method=method),
                self.concept_map, self.risk_tiers)
            for name, tm in model.task_models.items()
        }
        logger.info("MultiTaskInferenceService ready: %d tasks, %d windows",
                    len(self.builders), len(self.records))

    @classmethod
    def from_config(cls, cfg: Config, model_dir: str | None = None) -> "MultiTaskInferenceService":
        model = MultiTaskModel.load(model_dir or cfg.get("paths.model_dir", "artifacts/models_mimic"))
        events_dir = Path(cfg.get("data.mimic.events_path")).parent
        records_by_window = {}
        for w in _windows_used():
            anchor = _anchor_task(w)
            records_by_window[w] = _load_event_label_records(
                str(events_dir / T.events_filename(w)),
                str(events_dir / f"labels_{anchor.name}.parquet"), "mimic")
        return cls(cfg, model, records_by_window)

    # ----- public API --------------------------------------------------------
    def profile_for(self, patient_id: str) -> HealthRiskProfile:
        return self._build_profile(*self._inputs(patient_id))

    def profile_and_features(self, patient_id: str) -> tuple[HealthRiskProfile, dict]:
        """Profile + the featurized input, featurizing the patient only once."""
        rec, x_by_window = self._inputs(patient_id)
        return self._build_profile(rec, x_by_window), _feature_dict(x_by_window)

    def features_for(self, patient_id: str) -> dict:
        _, x_by_window = self._inputs(patient_id)
        return _feature_dict(x_by_window)

    def raw_ehr_payload_for(self, patient_id: str) -> dict:
        """Readable raw clinical values for the LLM baseline (M5): the SAME
        discharge-window features the pipeline uses, translated to plain names, with NO
        model prediction, score, or attribution attached."""
        rec, x_by_window = self._inputs(patient_id)
        row = x_by_window[_SNAPSHOT_WINDOW].iloc[0]
        concepts = mimic_concept_map()
        vitals: dict[str, float] = {}
        labs: dict[str, float] = {}
        history: list[str] = []
        prior_admissions = None
        for col, val in row.items():
            if pd.isna(val):
                continue
            code = col[len("value__"):] if col.startswith("value__") else col
            name = concepts[code].name if code in concepts else code
            family = code.split("/", 1)[0]
            if family == "VITAL" and col.startswith("value__"):
                vitals[name] = round(float(val), 2)
            elif family == "LAB" and col.startswith("value__"):
                labs[name] = round(float(val), 2)
            elif family == "DX" and float(val) >= 1 and name not in history:
                history.append(name)
            elif family == "UTIL":
                prior_admissions = round(float(val), 1)
        return {
            "demographics": {"age": rec.demographics.get("age"), "sex": rec.demographics.get("sex")},
            "vital_signs": vitals,
            "lab_results": labs,
            "past_conditions": history,
            "prior_admissions": prior_admissions,
            "note": "Raw recorded clinical values. No model prediction or score is provided.",
        }

    def any_patient_id(self) -> str:
        return next(iter(self.records[_SNAPSHOT_WINDOW].values())).patient_id

    # ----- internals ---------------------------------------------------------
    def _inputs(self, patient_id: str):
        """Return (snapshot record, {window: featurized 1-row DataFrame})."""
        pid = str(patient_id)
        x_by_window = {
            w: self.model.featurizer_for(w).transform([recs[pid]])
            for w, recs in self.records.items()
        }
        return self.records[_SNAPSHOT_WINDOW][pid], x_by_window

    def _build_profile(self, rec, x_by_window: dict) -> HealthRiskProfile:
        snapshot = build_snapshot(rec, self.lookback)
        forward: list[TaskPrediction] = []
        chronic: list[TaskPrediction] = []
        for name, tm in self.model.task_models.items():
            x_cols = x_by_window[tm.spec.window][tm.columns]
            out = tm.model.predict_output(x_cols)[0]
            builder = self.builders[name]
            if tm.spec.kind == "regression":
                tp = self._regression_prediction(tm, out, x_cols, builder)
            else:
                task_meta = SimpleNamespace(
                    name=tm.spec.name, description=tm.spec.label,
                    positive_label=tm.spec.positive_label, horizon=tm.spec.horizon,
                )
                rp = builder.build(out, x_cols, snapshot, task_meta, self.top_k)
                tp = TaskPrediction.from_risk_profile(rp, tm.spec, float(tm.metrics.get("auroc", 0.0)))
            (forward if tm.spec.group == "forward" else chronic).append(tp)

        by_prob = lambda t: t.probability if t.probability is not None else -1.0
        forward.sort(key=by_prob, reverse=True)
        chronic.sort(key=by_prob, reverse=True)
        return HealthRiskProfile(
            forward=forward, chronic=chronic, snapshot=snapshot.to_dict(),
            demographics={"age": rec.demographics.get("age"), "sex": rec.demographics.get("sex")},
            attribution_method=self._attribution_method(), notes=[],
        )

    def _attribution_method(self) -> str:
        # All task attributors share one method (shap or approx).
        return next(iter(self.builders.values())).attributor.method if self.builders else "approx"

    def _regression_prediction(self, tm, out, x_cols, builder) -> TaskPrediction:
        contribs = builder.attributor.explain(x_cols, self.top_k)
        max_mag = max((abs(c.signed_impact) for c in contribs), default=1.0) or 1.0
        return TaskPrediction.regression(
            tm.spec,
            point_estimate=float(out.point_estimate) if out.point_estimate is not None else None,
            uncertainty=float(out.uncertainty),
            confidence_label=RiskProfileBuilder.confidence_label(out.uncertainty),
            contributors=[builder.to_view(c, max_mag) for c in contribs if c.signed_impact > 0],
            protective_factors=[builder.to_view(c, max_mag) for c in contribs if c.signed_impact < 0],
        )


def _feature_dict(x_by_window: dict) -> dict:
    """Featurized model input per window (for response logging), keyed by tag."""
    out = {}
    for window, x in x_by_window.items():
        row = x.iloc[0]
        out[T.window_tag(window)] = {k: (None if pd.isna(v) else round(float(v), 4))
                                     for k, v in row.items()}
    return out
