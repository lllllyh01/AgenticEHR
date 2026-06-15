"""Map technical feature names to plain-language clinical concepts.

Feature names look like ``count__ICD10/I50.9`` or ``value__LAB/HbA1c``. This
maps them to (concept, patient-friendly phrase, direction-of-concern). The
built-in map covers the synthetic demo vocabulary; a YAML override can extend
it for real EHR-shot vocabularies without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Concept:
    name: str                 # short clinical concept, e.g. "Heart failure"
    patient_phrase: str       # plain-language phrase for a non-expert
    higher_is_concerning: bool = True  # does a higher feature value raise risk?


# Keyed by the *code* portion (after the count__/value__ prefix).
_BUILTIN: dict[str, Concept] = {
    "ICD10/E11.9": Concept("Type 2 diabetes", "a history of type 2 diabetes"),
    "ICD10/I50.9": Concept("Heart failure", "a history of heart failure"),
    "ICD10/J44.9": Concept("COPD", "a history of COPD (a chronic lung condition)"),
    "ICD10/N18.3": Concept("Chronic kidney disease", "reduced kidney function on record"),
    "ICD10/I10": Concept("Hypertension", "high blood pressure on record"),
    "ICD10/F32.9": Concept("Depression", "a history of depression"),
    "ICD10/Z79.4": Concept("Long-term insulin use", "long-term insulin use"),
    "ENC/INPATIENT": Concept("Prior hospital admissions", "recent hospital admissions"),
    "ENC/ED": Concept("Emergency visits", "recent emergency-department visits"),
    "LAB/HbA1c": Concept("HbA1c (blood sugar control)", "your recent blood-sugar (HbA1c) results"),
    "LAB/eGFR": Concept("Kidney function (eGFR)", "your recent kidney-function (eGFR) results",
                        higher_is_concerning=False),
    "MED/STATIN": Concept("Statin therapy", "being on statin therapy", higher_is_concerning=False),
    "MED/BETA_BLOCKER": Concept("Beta blocker", "being on a beta blocker", higher_is_concerning=False),
    # Non-code features.
    "age": Concept("Age", "your age"),
    "sex_female": Concept("Sex", "sex recorded as female"),
    "n_events": Concept("Overall care activity", "how much recent healthcare activity you've had"),
}


class ConceptMap:
    def __init__(self, mapping: dict[str, Concept] | None = None):
        self._map = dict(_BUILTIN)
        if mapping:
            self._map.update(mapping)

    @classmethod
    def from_config(cls, cfg) -> "ConceptMap":
        path = cfg.get("explain.concept_map_path")
        if not path:
            return cls()
        return cls(_load_override(path))

    def resolve(self, feature_name: str) -> Concept:
        code = _strip_prefix(feature_name)
        if code in self._map:
            return self._map[code]
        if feature_name in self._map:
            return self._map[feature_name]
        # Unknown code: degrade gracefully to a neutral, honest phrasing.
        return Concept(
            name=code,
            patient_phrase=f"a clinical factor recorded as '{code}'",
        )


def _strip_prefix(feature_name: str) -> str:
    for prefix in ("count__", "value__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name


def _load_override(path: str | Path) -> dict[str, Concept]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"concept_map_path not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, Concept] = {}
    for code, spec in raw.items():
        out[code] = Concept(
            name=spec["name"],
            patient_phrase=spec.get("patient_phrase", spec["name"]),
            higher_is_concerning=spec.get("higher_is_concerning", True),
        )
    logger.info("Loaded %d concept-map overrides from %s", len(out), path)
    return out
