"""Clinical concept dictionaries for MIMIC-IV.

Design (health-summary agent):
  * Prediction TARGETS  : chronic phenotypes (ICD groups) + forward outcomes.
  * Shared model INPUT   : vital-sign summaries + a lab panel + prior-admission
    comorbidity history + utilization + demographics, all observed up to the
    discharge anchor and built by one featurizer (train == inference).

The sets below are pragmatic defaults aligned with common MIMIC-IV phenotyping
practice and SHOULD be reviewed by a clinician before any real analysis.

Emitted feature codes (consumed by the FEMR-style CountFeaturizer):
  VITAL/<NAME>_<STAT>  vital-sign summary statistic over the admission
  LAB/<NAME>           latest lab value in the admission
  DX/<GROUP>           comorbidity present in PRIOR admissions (history feature)
  UTIL/<NAME>          utilization count (e.g. prior admissions)
"""
from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# ICD diagnosis groups — used BOTH as chronic-phenotype labels and as          #
# prior-admission comorbidity-history features.                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IcdGroup:
    code: str
    description: str
    icd9_prefixes: tuple[str, ...] = ()
    icd10_prefixes: tuple[str, ...] = ()
    icd9_range: tuple[int, int] | None = None


ICD_GROUPS: tuple[IcdGroup, ...] = (
    IcdGroup("DIABETES", "Diabetes mellitus",
             icd9_prefixes=("250",),
             icd10_prefixes=("E08", "E09", "E10", "E11", "E13")),
    IcdGroup("HYPERTENSION", "Hypertension",
             icd9_prefixes=("401", "402", "403", "404", "405"),
             icd10_prefixes=("I10", "I11", "I12", "I13", "I15", "I16")),
    IcdGroup("HYPERLIPIDEMIA", "Hyperlipidemia / dyslipidemia",
             icd9_prefixes=("272",), icd10_prefixes=("E78",)),
    IcdGroup("CARDIOVASCULAR", "Cardiovascular disease (ischemic, heart failure, cerebrovascular)",
             icd9_prefixes=("410", "411", "412", "413", "414", "428",
                            "430", "431", "432", "433", "434", "435", "436", "437", "438"),
             icd10_prefixes=("I20", "I21", "I22", "I23", "I24", "I25", "I50",
                             "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69")),
    IcdGroup("RESPIRATORY", "Respiratory disease",
             icd9_range=(460, 519), icd10_prefixes=("J",)),
    IcdGroup("DEPRESSION_ANXIETY", "Depression or anxiety disorder",
             icd9_prefixes=("2962", "2963", "311", "3004", "3000", "30002"),
             icd10_prefixes=("F32", "F33", "F34", "F40", "F41")),
)

# Chronic-phenotype prediction targets (label code -> description).
CHRONIC_TARGETS: tuple[IcdGroup, ...] = ICD_GROUPS


def _normalize_icd(code: str) -> str:
    return str(code).replace(".", "").strip().upper()


def classify_icd(icd_code: str, icd_version: int | str) -> list[str]:
    """Return the group codes (e.g. ``["DIABETES"]``) an ICD code maps to."""
    norm = _normalize_icd(icd_code)
    try:
        version = int(icd_version)
    except (TypeError, ValueError):
        version = 0
    hits: list[str] = []
    for group in ICD_GROUPS:
        if version == 9:
            if any(norm.startswith(p) for p in group.icd9_prefixes):
                hits.append(group.code)
                continue
            if group.icd9_range is not None and len(norm) >= 3 and norm[:3].isdigit():
                lo, hi = group.icd9_range
                if lo <= int(norm[:3]) <= hi:
                    hits.append(group.code)
        elif version == 10:
            if any(norm.startswith(p) for p in group.icd10_prefixes):
                hits.append(group.code)
    return hits


# --------------------------------------------------------------------------- #
# Vital-sign time series (icu/chartevents) -> summary-statistic features        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VitalConcept:
    code: str
    description: str
    itemids: tuple[int, ...]
    fahrenheit_itemids: tuple[int, ...] = ()   # converted to Celsius before aggregation


VITAL_SERIES: tuple[VitalConcept, ...] = (
    VitalConcept("HR", "Heart rate", (220045,)),
    VitalConcept("RR", "Respiratory rate", (220210,)),
    VitalConcept("SPO2", "Oxygen saturation (SpO2)", (220277,)),
    VitalConcept("TEMP_C", "Body temperature", (223762,), fahrenheit_itemids=(223761,)),
    VitalConcept("SBP", "Systolic blood pressure", (220179, 220050)),
    VitalConcept("DBP", "Diastolic blood pressure", (220180, 220051)),
    VitalConcept("MBP", "Mean arterial pressure", (220181, 220052)),
)

# Summary statistics computed per vital over the observation (admission) window.
VITAL_STATS: tuple[str, ...] = ("mean", "min", "max", "last", "std")


# --------------------------------------------------------------------------- #
# Lab panel (hosp/labevents) -> latest-value features                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NumericConcept:
    code: str
    description: str
    itemids: tuple[int, ...]
    unit: str = ""


LAB_PANEL: tuple[NumericConcept, ...] = (
    NumericConcept("GLUCOSE", "Glucose", (50931,), "mg/dL"),
    NumericConcept("HBA1C", "Hemoglobin A1c", (50852,), "%"),
    NumericConcept("TRIGLYCERIDES", "Triglycerides", (51000,), "mg/dL"),
    NumericConcept("CHOLESTEROL", "Total cholesterol", (50907,), "mg/dL"),
    NumericConcept("CREATININE", "Creatinine", (50912,), "mg/dL"),
    NumericConcept("UREA_NITROGEN", "Blood urea nitrogen", (51006,), "mg/dL"),
    NumericConcept("SODIUM", "Sodium", (50983,), "mEq/L"),
    NumericConcept("POTASSIUM", "Potassium", (50971,), "mEq/L"),
    NumericConcept("BICARBONATE", "Bicarbonate", (50882,), "mEq/L"),
    NumericConcept("CHLORIDE", "Chloride", (50902,), "mEq/L"),
    NumericConcept("HEMOGLOBIN", "Hemoglobin", (51222,), "g/dL"),
    NumericConcept("WBC", "White blood cell count", (51301,), "K/uL"),
    NumericConcept("PLATELET", "Platelet count", (51265,), "K/uL"),
    NumericConcept("ALBUMIN", "Albumin", (50862,), "g/dL"),
    NumericConcept("ALT", "Alanine aminotransferase (ALT)", (50861,), "IU/L"),
    NumericConcept("AST", "Aspartate aminotransferase (AST)", (50878,), "IU/L"),
    NumericConcept("BILIRUBIN", "Total bilirubin", (50885,), "mg/dL"),
)

FAHRENHEIT_TO_CELSIUS = "(valuenum - 32.0) / 1.8"


# --------------------------------------------------------------------------- #
# Per-target feature exclusions ("don't feed the literal answer").             #
# When training a chronic-phenotype target, drop feature columns whose code    #
# matches any of these prefixes, so the task is not a no-op.                    #
# --------------------------------------------------------------------------- #
DX_DEFINING_FEATURES: dict[str, tuple[str, ...]] = {
    "DIABETES": ("DX/DIABETES", "LAB/HBA1C", "LAB/GLUCOSE"),
    "HYPERTENSION": ("DX/HYPERTENSION",),
    "HYPERLIPIDEMIA": ("DX/HYPERLIPIDEMIA", "LAB/CHOLESTEROL", "LAB/TRIGLYCERIDES"),
    "CARDIOVASCULAR": ("DX/CARDIOVASCULAR",),
    "RESPIRATORY": ("DX/RESPIRATORY",),
    "DEPRESSION_ANXIETY": ("DX/DEPRESSION_ANXIETY",),
}
