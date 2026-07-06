"""The RiskProfile: the single contract between models and the agent.

A ``RiskProfile`` is a plain, serialisable object. Anything that can produce one
(XGBoost today, MOTOR-T or a foundation model tomorrow) is a valid backend for
the agent. The agent reads ONLY this object plus the patient snapshot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..data.schema import PatientSnapshot, TaskMetadata
from .attributions import Attributor, Contribution
from .concept_map import ConceptMap


@dataclass
class ContributorView:
    """A contributor expressed in plain language for the agent."""

    concept: str               # e.g. "Heart failure"
    patient_phrase: str        # e.g. "a history of heart failure"
    direction: str             # "increases" | "decreases"
    magnitude: float           # |signed impact|, relative within this profile
    observed_value: float
    source_feature: str
    method: str


@dataclass
class RiskProfile:
    task: dict[str, str]
    probability: float                 # calibrated probability of positive outcome
    raw_probability: float
    risk_tier: str                     # "low" | "moderate" | "elevated"
    uncertainty: float                 # [0,1], higher = less confident
    confidence_label: str              # "lower" | "moderate" | "higher" confidence
    contributors: list[ContributorView]
    protective_factors: list[ContributorView]
    snapshot: dict[str, Any]
    attribution_method: str            # "shap" | "approx"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def probability_pct(self) -> int:
        return int(round(self.probability * 100))


@dataclass
class TaskPrediction:
    """One label's prediction within a multi-task health profile.

    Binary tasks populate the probability/risk_tier/auroc fields; regression tasks
    leave them ``None`` and populate ``point_estimate`` instead.
    """

    name: str
    label: str
    group: str                         # "forward" | "chronic"
    kind: str                          # "binary" | "regression"
    positive_label: str
    horizon: str
    uncertainty: float
    confidence_label: str
    contributors: list[ContributorView]
    protective_factors: list[ContributorView]
    probability: float | None = None        # binary only
    raw_probability: float | None = None     # binary only
    risk_tier: str | None = None             # binary only
    auroc: float | None = None               # held-out test AUROC (binary only)
    point_estimate: float | None = None      # regression only

    @property
    def probability_pct(self) -> int | None:
        return None if self.probability is None else int(round(self.probability * 100))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_risk_profile(cls, rp: "RiskProfile", spec, auroc: float) -> "TaskPrediction":
        """Build a binary task prediction from a single-task RiskProfile.

        ``spec`` is duck-typed (needs ``name``/``label``/``group``/
        ``positive_label``/``horizon``), so this stays free of MIMIC imports.
        """
        return cls(
            name=spec.name, label=spec.label, group=spec.group, kind="binary",
            positive_label=spec.positive_label, horizon=spec.horizon,
            uncertainty=rp.uncertainty, confidence_label=rp.confidence_label,
            contributors=rp.contributors, protective_factors=rp.protective_factors,
            probability=rp.probability, raw_probability=rp.raw_probability,
            risk_tier=rp.risk_tier, auroc=auroc,
        )

    @classmethod
    def regression(cls, spec, *, point_estimate, uncertainty, confidence_label,
                   contributors, protective_factors) -> "TaskPrediction":
        return cls(
            name=spec.name, label=spec.label, group=spec.group, kind="regression",
            positive_label=spec.positive_label, horizon=spec.horizon,
            uncertainty=uncertainty, confidence_label=confidence_label,
            contributors=contributors, protective_factors=protective_factors,
            point_estimate=point_estimate,
        )


@dataclass
class HealthRiskProfile:
    """Multi-label prediction panel — the contract the health-summary agent reads.

    Mirrors :class:`RiskProfile` but carries a panel of :class:`TaskPrediction`
    (forward-looking risks + chronic-phenotype profile) sharing one patient
    snapshot, instead of a single prediction.
    """

    forward: list[TaskPrediction]
    chronic: list[TaskPrediction]
    snapshot: dict[str, Any]
    demographics: dict[str, Any]
    attribution_method: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskProfileBuilder:
    """Assembles a :class:`RiskProfile` from a model + attributions + concepts.

    This is the *adapter* layer. Swapping the model means re-pointing this
    builder; the resulting RiskProfile (and thus the agent) is unchanged.
    """

    def __init__(self, attributor: Attributor, concept_map: ConceptMap, risk_tiers: list[dict]):
        self.attributor = attributor
        self.concept_map = concept_map
        self.risk_tiers = risk_tiers

    def build(
        self,
        model_output,                  # ModelOutput
        x_row,                         # 1-row DataFrame of features
        snapshot: PatientSnapshot,
        task: TaskMetadata,
        top_k: int = 5,
    ) -> RiskProfile:
        contribs = self.attributor.explain(x_row, top_k=top_k)
        increasing = [c for c in contribs if c.signed_impact > 0]
        decreasing = [c for c in contribs if c.signed_impact < 0]

        max_mag = max((abs(c.signed_impact) for c in contribs), default=1.0) or 1.0
        views_up = [self.to_view(c, max_mag) for c in increasing]
        views_down = [self.to_view(c, max_mag) for c in decreasing]

        tier = self._tier(model_output.probability)
        conf_label = self.confidence_label(model_output.uncertainty)

        notes = []
        if self.attributor.method == "approx":
            notes.append(
                "Contributor estimates are approximate (SHAP not installed); "
                "treat the ordering as indicative, not exact."
            )

        return RiskProfile(
            task={
                "name": task.name,
                "description": task.description,
                "positive_label": task.positive_label,
                "horizon": task.horizon,
            },
            probability=float(model_output.probability),
            raw_probability=float(model_output.raw_probability),
            risk_tier=tier,
            uncertainty=float(model_output.uncertainty),
            confidence_label=conf_label,
            contributors=views_up,
            protective_factors=views_down,
            snapshot=snapshot.to_dict(),
            attribution_method=self.attributor.method,
            notes=notes,
        )

    def to_view(self, c: Contribution, max_mag: float) -> ContributorView:
        concept = self.concept_map.resolve(c.feature)
        return ContributorView(
            concept=concept.name,
            patient_phrase=concept.patient_phrase,
            direction="increases" if c.signed_impact > 0 else "decreases",
            magnitude=round(abs(c.signed_impact) / max_mag, 3),
            observed_value=round(c.value, 3),
            source_feature=c.feature,
            method=c.method,
        )

    def _tier(self, prob: float) -> str:
        for rule in self.risk_tiers:
            if prob < rule["max"]:
                return rule["tier"]
        return self.risk_tiers[-1]["tier"]

    @staticmethod
    def confidence_label(uncertainty: float) -> str:
        if uncertainty < 0.34:
            return "higher"
        if uncertainty < 0.67:
            return "moderate"
        return "lower"
