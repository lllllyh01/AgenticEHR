"""Reproducible synthetic EHR generator.

Produces longitudinal coded events with a *learnable* relationship to the
label, so the XGBoost baseline has real signal to fit. No PHI, fully seeded.

The generative story (transparent on purpose): a latent risk score is built
from a handful of clinically plausible drivers (age, diabetes, heart failure,
high HbA1c, prior admissions, ...). The label is sampled from that score.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .schema import Event, PatientRecord

# Codes the generator can emit, with a "risk weight" used to build the latent
# score. Positive weight -> raises risk; the model must rediscover these.
_CODE_LIBRARY = {
    "ICD10/E11.9": ("Type 2 diabetes", 0.9),
    "ICD10/I50.9": ("Heart failure", 1.3),
    "ICD10/J44.9": ("COPD", 1.0),
    "ICD10/N18.3": ("Chronic kidney disease", 1.1),
    "ICD10/I10": ("Hypertension", 0.5),
    "ICD10/F32.9": ("Depression", 0.3),
    "ICD10/Z79.4": ("Long-term insulin use", 0.6),
    "ENC/INPATIENT": ("Inpatient admission", 1.2),
    "ENC/ED": ("Emergency visit", 0.8),
    "LAB/HbA1c": ("HbA1c level", 0.0),        # value-driven, see below
    "LAB/eGFR": ("Kidney function (eGFR)", 0.0),
    "MED/STATIN": ("Statin therapy", -0.3),   # protective
    "MED/BETA_BLOCKER": ("Beta blocker", -0.2),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_records(
    n_patients: int = 4000,
    positive_rate: float = 0.18,
    seed: int = 42,
    end_date: datetime | None = None,
) -> list[PatientRecord]:
    """Generate ``n_patients`` synthetic :class:`PatientRecord` objects."""
    rng = np.random.default_rng(seed)
    end_date = end_date or datetime(2020, 1, 1)
    records: list[PatientRecord] = []

    codes = list(_CODE_LIBRARY)
    for i in range(n_patients):
        age = int(np.clip(rng.normal(62, 16), 18, 98))
        sex = "female" if rng.random() < 0.51 else "male"
        prediction_time = end_date

        events: list[Event] = []
        latent = 0.02 * (age - 60)  # age contributes to latent risk

        # Each chronic/acute code present with some probability; if present,
        # emit one or more timestamped events and add its weight.
        for code in codes:
            name, weight = _CODE_LIBRARY[code]
            base_p = {
                "ICD10/I10": 0.45, "ICD10/E11.9": 0.30, "ICD10/F32.9": 0.20,
            }.get(code, 0.15)
            if rng.random() < base_p:
                n_occ = 1 + rng.poisson(1.0)
                for _ in range(n_occ):
                    days_ago = int(rng.integers(1, 360))
                    value = None
                    if code == "LAB/HbA1c":
                        value = float(np.clip(rng.normal(6.5, 1.6), 4.5, 14))
                        latent += 0.35 * max(0.0, value - 7.0)
                    elif code == "LAB/eGFR":
                        value = float(np.clip(rng.normal(75, 22), 8, 120))
                        latent += 0.02 * max(0.0, 60 - value)
                    events.append(
                        Event(
                            time=prediction_time - timedelta(days=days_ago),
                            code=code,
                            value=value,
                            description=name,
                        )
                    )
                latent += weight * (1 if code.startswith(("ICD", "MED")) else min(n_occ, 4) * 0.5)

        # Demographics row.
        demographics = {"age": age, "sex": sex}

        # Convert latent score to a probability, then sample the label. The
        # intercept is tuned so the overall positive rate ~ positive_rate.
        records.append(
            PatientRecord(
                patient_id=f"SYN{i:06d}",
                events=sorted(events, key=lambda e: e.time),
                prediction_time=prediction_time,
                label=None,            # filled after we calibrate the intercept
                demographics=demographics,
            )
        )
        records[-1]._latent = latent  # type: ignore[attr-defined]

    # Calibrate intercept to hit the requested positive rate, then sample.
    latents = np.array([r._latent for r in records])  # type: ignore[attr-defined]
    intercept = _solve_intercept(latents, positive_rate)
    probs = _sigmoid(latents + intercept)
    labels = (rng.random(len(records)) < probs).astype(int)
    for r, y in zip(records, labels):
        r.label = int(y)
        delattr(r, "_latent")
    return records


def _solve_intercept(latents: np.ndarray, target_rate: float) -> float:
    """Binary-search an intercept so mean sigmoid(latent + b) == target_rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = float(_sigmoid(latents + mid).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
