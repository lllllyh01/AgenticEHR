"""Core data structures shared across the pipeline.

These deliberately mirror the shape of FEMR / EHR-shot data (a patient is a
time-ordered sequence of coded events, plus a label anchored at a prediction
time) without depending on FEMR itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """A single coded clinical event in a patient's timeline."""

    time: datetime
    code: str                     # e.g. "ICD10/E11.9", "LAB/HbA1c"
    value: float | str | None = None
    description: str | None = None  # optional human-readable label


@dataclass
class PatientRecord:
    """A patient's full longitudinal record plus the labelled prediction point."""

    patient_id: str
    events: list[Event]
    prediction_time: datetime
    label: int | float | None = None  # ground-truth outcome (float for regression; None at inference)
    demographics: dict[str, Any] = field(default_factory=dict)

    def events_before(self, time: datetime) -> list[Event]:
        return [e for e in self.events if e.time <= time]


@dataclass
class TaskMetadata:
    """Describes the prediction task so the agent can speak about it correctly."""

    name: str
    description: str
    positive_label: str           # plain-language meaning of "y = 1"
    horizon: str                  # e.g. "the next 12 months"

    @classmethod
    def from_config(cls, cfg) -> "TaskMetadata":
        return cls(
            name=cfg.get("task.name", "risk_task"),
            description=cfg.get("task.description", "a health risk"),
            positive_label=cfg.get("task.positive_label", "the outcome"),
            horizon=cfg.get("task.horizon", "the coming months"),
        )


@dataclass
class PatientSnapshot:
    """A compact, agent-facing view of the patient's *current* state.

    This is intentionally small and PHI-light: the agent only needs salient,
    non-identifying clinical context to phrase a summary, never raw identifiers.
    """

    patient_id: str
    age: int | None = None
    sex: str | None = None
    active_conditions: list[str] = field(default_factory=list)
    recent_observations: dict[str, Any] = field(default_factory=dict)
    n_recent_encounters: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "sex": self.sex,
            "active_conditions": self.active_conditions,
            "recent_observations": self.recent_observations,
            "n_recent_encounters": self.n_recent_encounters,
        }
