"""Multi-task inference service: patient -> HealthRiskProfile (prediction panel).

Loads the trained per-task models and, for one patient, predicts every task and
attributes each prediction, assembling the multi-label ``HealthRiskProfile`` the
health-summary agent consumes.
"""
from __future__ import annotations

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
from .multitask import MultiTaskModel

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


class MultiTaskInferenceService:
    def __init__(self, cfg: Config, model: MultiTaskModel, records):
        self.cfg = cfg
        self.model = model
        self.records = {r.patient_id: r for r in records}
        self.lookback = cfg.get("data.featurize.lookback_days", 3650)
        self.top_k = cfg.get("explain.top_k_contributors", 5)
        self.concept_map = ConceptMap(mimic_concept_map())
        self.risk_tiers = cfg.get("agent.risk_tiers")
        method = cfg.get("explain.method", "auto")

        # Background (for attribution medians) = the full feature matrix.
        background = model.featurizer.transform(records)
        self.builders: dict[str, RiskProfileBuilder] = {}
        for name, tm in model.task_models.items():
            attributor = Attributor(tm.model, background[tm.columns], method=method)
            self.builders[name] = RiskProfileBuilder(attributor, self.concept_map, self.risk_tiers)
        logger.info("MultiTaskInferenceService ready: %d tasks, %d patients",
                    len(self.builders), len(self.records))

    @classmethod
    def from_config(cls, cfg: Config, model_dir: str | None = None) -> "MultiTaskInferenceService":
        model = MultiTaskModel.load(model_dir or cfg.get("paths.model_dir", "artifacts/models_mimic"))
        events_path = cfg.get("data.mimic.events_path")
        anchor_labels = str(__import__("pathlib").Path(events_path).parent / f"labels_{T.ALL_TASKS[0].name}.parquet")
        records = _load_event_label_records(events_path, anchor_labels, "mimic")
        return cls(cfg, model, records)

    def profile_for(self, patient_id: str) -> HealthRiskProfile:
        rec = self.records[str(patient_id)]
        x_full = self.model.featurizer.transform([rec])
        snapshot = build_snapshot(rec, self.lookback)

        forward: list[TaskPrediction] = []
        chronic: list[TaskPrediction] = []
        method = "approx"
        for name, tm in self.model.task_models.items():
            x_cols = x_full[tm.columns]
            out = tm.model.predict_output(x_cols)[0]
            builder = self.builders[name]
            method = builder.attributor.method
            if tm.spec.kind == "regression":
                tp = self._regression_prediction(tm, out, x_cols, builder)
            else:
                task_meta = SimpleNamespace(
                    name=tm.spec.name, description=tm.spec.label,
                    positive_label=tm.spec.positive_label, horizon=tm.spec.horizon,
                )
                rp = builder.build(out, x_cols, snapshot, task_meta, self.top_k)
                tp = TaskPrediction(
                    name=tm.spec.name, label=tm.spec.label, group=tm.spec.group, kind=tm.spec.kind,
                    positive_label=tm.spec.positive_label, horizon=tm.spec.horizon,
                    probability=rp.probability, raw_probability=rp.raw_probability,
                    risk_tier=rp.risk_tier, uncertainty=rp.uncertainty,
                    confidence_label=rp.confidence_label, auroc=float(tm.metrics.get("auroc", 0.0)),
                    contributors=rp.contributors, protective_factors=rp.protective_factors,
                )
            (forward if tm.spec.group == "forward" else chronic).append(tp)

        forward.sort(key=lambda t: t.probability, reverse=True)
        chronic.sort(key=lambda t: t.probability, reverse=True)
        return HealthRiskProfile(
            forward=forward, chronic=chronic,
            snapshot=snapshot.to_dict(),
            demographics={"age": rec.demographics.get("age"), "sex": rec.demographics.get("sex")},
            attribution_method=method,
            notes=[],
        )

    def _regression_prediction(self, tm, out, x_cols, builder) -> TaskPrediction:
        contribs = builder.attributor.explain(x_cols, self.top_k)
        max_mag = max((abs(c.signed_impact) for c in contribs), default=1.0) or 1.0
        views_up = [builder._to_view(c, max_mag) for c in contribs if c.signed_impact > 0]
        views_down = [builder._to_view(c, max_mag) for c in contribs if c.signed_impact < 0]
        return TaskPrediction(
            name=tm.spec.name, label=tm.spec.label, group=tm.spec.group, kind="regression",
            positive_label=tm.spec.positive_label, horizon=tm.spec.horizon,
            probability=0.0, raw_probability=0.0, risk_tier="n/a",
            uncertainty=float(out.uncertainty),
            confidence_label=RiskProfileBuilder._confidence_label(out.uncertainty),
            auroc=0.0, contributors=views_up, protective_factors=views_down,
            point_estimate=float(out.point_estimate) if out.point_estimate is not None else None,
        )

    def features_for(self, patient_id: str) -> dict:
        """The featurized model input for one patient (for response logging)."""
        row = self.model.featurizer.transform([self.records[str(patient_id)]]).iloc[0]
        return {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()}

    def any_patient_id(self) -> str:
        return next(iter(self.records))
