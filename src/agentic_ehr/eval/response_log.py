"""Persist an inference (metadata + predictions + agent report) for review.

Writes one record per patient as JSON / Markdown / TXT, with the model logits,
the EHR feature input, and the predictions placed ABOVE the report — so a
clinician can inspect the reasoning behind the model's inference and evaluate the
agent's output. Works for both a single-task ``RiskProfile`` and a multi-label
``HealthRiskProfile``.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..explain.risk_profile import HealthRiskProfile


def _logit(p: float) -> float | None:
    if p is None or p <= 0.0 or p >= 1.0:
        return None
    return round(math.log(p / (1.0 - p)), 4)


def _contributors(views) -> list[dict]:
    return [
        {"concept": c.concept, "phrase": c.patient_phrase, "direction": c.direction,
         "magnitude": c.magnitude, "observed_value": c.observed_value, "feature": c.source_feature}
        for c in views
    ]


def _prediction(tp) -> dict:
    """Serialise one TaskPrediction (binary or regression)."""
    d = {
        "task": tp.name, "label": tp.label, "group": tp.group, "kind": tp.kind,
        "confidence": tp.confidence_label, "uncertainty": round(tp.uncertainty, 4),
        "attribution_method": tp.contributors[0].method if tp.contributors else None,
        "top_contributors": _contributors(tp.contributors[:5]),
    }
    if tp.kind == "regression":
        d["point_estimate"] = tp.point_estimate
    else:
        d.update({"probability": round(tp.probability, 4), "probability_pct": tp.probability_pct,
                  "logit": _logit(tp.raw_probability), "raw_probability": round(tp.raw_probability, 4),
                  "risk_tier": tp.risk_tier, "auroc": tp.auroc})
    return d


def build_record(patient_id, profile, summary, *, model_info: dict | None = None,
                 features: dict | None = None) -> dict:
    meta = {
        "patient_id": str(patient_id),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_backend": summary.backend,
        "mode": summary.mode,
        "model": model_info or {},
    }
    if profile is None:  # LLM baseline: raw EHR fed directly, no prediction model
        predictions = {"note": "LLM baseline — raw EHR fed directly to the model; "
                               "no prediction model, score, or attribution."}
        snapshot = None
        demographics = (features or {}).get("demographics", {})
    elif isinstance(profile, HealthRiskProfile):
        predictions = {
            "forward": [_prediction(t) for t in profile.forward],
            "chronic": [_prediction(t) for t in profile.chronic],
        }
        snapshot = profile.snapshot
        demographics = profile.demographics
    else:  # single-task RiskProfile
        predictions = {
            "task": profile.task,
            "probability": round(profile.probability, 4),
            "probability_pct": profile.probability_pct,
            "logit": _logit(profile.raw_probability),
            "raw_probability": round(profile.raw_probability, 4),
            "risk_tier": profile.risk_tier, "confidence": profile.confidence_label,
            "uncertainty": round(profile.uncertainty, 4),
            "top_contributors": _contributors(profile.contributors[:5]),
            "protective_factors": _contributors(profile.protective_factors[:5]),
        }
        snapshot = profile.snapshot
        demographics = {"age": profile.snapshot.get("age"), "sex": profile.snapshot.get("sex")}

    return {
        "metadata": meta,
        "report": {"sections": summary.sections, "disclaimer": summary.disclaimer},
        "predictions": predictions,
        "input": {"demographics": demographics, "features": features, "snapshot": snapshot},
    }


def to_markdown(record: dict) -> str:
    m = record["metadata"]
    lines = [f"# Inference record — patient {m['patient_id']}", "",
             f"- generated_at: {m['generated_at']}",
             f"- agent_backend: {m['agent_backend']} | mode: {m['mode']}",
             f"- model: {json.dumps(m['model'])}", "",
             "## Agent report", ""]
    for sec, txt in record["report"]["sections"].items():
        lines += [f"### {sec}", txt, ""]
    lines += ["---", f"_{record['report']['disclaimer']}_", "",
              "## Predictions (model output — below the report)", "",
              "```json", json.dumps(record["predictions"], indent=2), "```", "",
              "## EHR feature input", "",
              "```json", json.dumps(record["input"], indent=2, default=str), "```"]
    return "\n".join(lines)


def to_text(record: dict) -> str:
    out = to_markdown(record)
    return out.replace("```json", "").replace("```", "").replace("# ", "").replace("#", "")


def write_record(record: dict, out_dir: str, formats=("json", "md"),
                 out_filename: str | None = None) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_filename = out_filename or record["metadata"]["patient_id"]
    renderers = {"json": lambda: json.dumps(record, indent=2, default=str),
                 "md": lambda: to_markdown(record),
                 "txt": lambda: to_text(record)}
    written = []
    for ext in ("json", "md", "txt"):
        if ext in formats:
            p = out / f"{out_filename}.{ext}"
            p.write_text(renderers[ext]())
            written.append(str(p))
    return written
