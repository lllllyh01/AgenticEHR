"""The SummaryAgent: drives an LLM backend and enforces safety guardrails.

The AI (LLM) summary agent is the core of the system — and the only summary
generator. It handles BOTH a single-task :class:`RiskProfile` (one prediction)
and a multi-label :class:`HealthRiskProfile` (a prediction panel), dispatching on
the profile type. There is **no fallback**: if the backend cannot run (missing
SDK / API key, or an API error) the agent RAISES so the failure is surfaced.
Safety-guardrail violations are **non-blocking** — they are recorded on the
returned summary (``guardrail_warnings``) but do not abort it. The backend is provider-agnostic
(Gemini / Claude / GPT) and pure transport; see :mod:`agent.llm_backend`.

The constructor accepts any object implementing the backend contract
(``name`` + ``run_sections(system_template, system_free, payload, use_template)``),
which keeps the agent decoupled from the provider and lets callers inject a test
double.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..explain.risk_profile import HealthRiskProfile, RiskProfile
from ..logging_utils import get_logger
from . import templates as T

logger = get_logger(__name__)


def _top_forward_risk(profile: HealthRiskProfile):
    """Highest-probability binary forward risk (regression tasks have no probability)."""
    binary = [t for t in profile.forward if t.probability is not None]
    return max(binary, key=lambda t: t.probability) if binary else None


def _single_payload(profile: RiskProfile) -> dict:
    return {**profile.to_dict(), "probability_pct": profile.probability_pct}


def _health_payload(profile: HealthRiskProfile) -> dict:
    """Compact, LLM-friendly view of a multi-label prediction panel."""
    return {
        "demographics": profile.demographics,
        "forward_risks": [
            {
                "outcome": t.label,
                "chance_percent": t.probability_pct,
                "confidence": t.confidence_label,
                "model_reliability": t.reliability,
                "horizon": t.horizon,
                "top_factors": [c.patient_phrase for c in t.contributors[:3]],
            }
            for t in profile.forward if t.kind == "binary"
        ],
        "expected_outcomes": [
            {"outcome": t.label, "estimate": round(t.point_estimate, 1),
             "confidence": t.confidence_label, "model_reliability": t.reliability,
             "top_factors": [c.patient_phrase for c in t.contributors[:3]]}
            for t in profile.forward if t.kind == "regression" and t.point_estimate is not None
        ],
        "chronic_profile": [
            {"condition": t.label, "likelihood_percent": t.probability_pct,
             "confidence": t.confidence_label, "model_reliability": t.reliability}
            for t in profile.chronic
        ],
        "recent_snapshot": profile.snapshot,
        "note": "All numbers are model estimates, not confirmed diagnoses.",
    }


class SummaryGuardrailError(RuntimeError):
    """Raised when a backend's summary fails the safety guardrails."""

    def __init__(self, warnings: list[str]):
        self.warnings = warnings
        super().__init__("Summary failed safety guardrails: " + "; ".join(warnings))


@dataclass
class PatientSummary:
    sections: dict[str, str]
    disclaimer: str
    backend: str
    risk_tier: str
    probability_pct: int
    mode: str = "template"  # "template" (five fixed sections) | "free" (narrative)
    guardrail_warnings: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        if self.mode == "free":
            body = self.sections.get("Summary", "").strip()
        else:
            body = "\n\n".join(
                f"## {sec}\n{self.sections.get(sec, '').strip()}" for sec in T.SECTIONS
            )
        return f"{body}\n\n---\n_{self.disclaimer}_"

    def to_dict(self) -> dict:
        return {
            "sections": self.sections,
            "disclaimer": self.disclaimer,
            "backend": self.backend,
            "mode": self.mode,
            "risk_tier": self.risk_tier,
            "probability_pct": self.probability_pct,
            "guardrail_warnings": self.guardrail_warnings,
        }


class SummaryAgent:
    def __init__(self, backend):
        if backend is None:
            raise ValueError("SummaryAgent requires an explicit backend.")
        self.backend = backend

    @classmethod
    def from_config(cls, cfg) -> "SummaryAgent":
        # Construct the LLM backend for the configured provider; let construction
        # errors propagate so a missing SDK/key surfaces immediately rather than
        # being masked. provider defaults to gemini (the best free model);
        # model=None picks the provider's best default.
        from .llm_backend import make_llm_backend
        backend = make_llm_backend(
            provider=cfg.get("agent.llm.provider", "gemini"),
            model=cfg.get("agent.llm.model", None),
            max_tokens=cfg.get("agent.llm.max_tokens", 4000),
        )
        return cls(backend)

    def summarize(self, profile, use_template: bool = True) -> PatientSummary:
        """Generate a summary from a RiskProfile or a HealthRiskProfile.

        ``use_template=True`` produces the five fixed sections (structured
        output); ``use_template=False`` produces one free-form narrative. Safety
        guardrails run in both modes but are non-blocking: any violations are
        attached to the returned summary's ``guardrail_warnings`` rather than
        raised, so a report is always produced.
        """
        if isinstance(profile, HealthRiskProfile):
            sections = self.backend.run_sections(
                T.REPORT_SYSTEM_PROMPT, T.REPORT_SYSTEM_PROMPT_FREE,
                _health_payload(profile), use_template,
            )
            warnings = self._validate_health(sections, profile, use_template)
            top = _top_forward_risk(profile)
            risk_tier = top.risk_tier if top else "n/a"
            probability_pct = top.probability_pct if top else 0
        else:
            sections = self.backend.run_sections(
                T.SINGLE_SYSTEM_PROMPT, T.SINGLE_SYSTEM_PROMPT_FREE,
                _single_payload(profile), use_template,
            )
            warnings = self._validate_single(sections, profile, use_template)
            risk_tier = profile.risk_tier
            probability_pct = profile.probability_pct

        # Non-blocking guardrails: violations are recorded on the summary and surfaced
        # to the caller, but do NOT raise — a report is always produced (parallels the LLM
        # baseline). The warning rate becomes a comparison signal rather than an abort.
        if warnings:
            logger.warning("Guardrail warnings from %s backend (not blocking): %s",
                           self.backend.name, warnings)

        return PatientSummary(
            sections=sections,
            disclaimer=T.DISCLAIMER,
            backend=self.backend.name,
            risk_tier=risk_tier,
            probability_pct=probability_pct,
            mode="template" if use_template else "free",
            guardrail_warnings=warnings,
        )

    # ----- guardrails --------------------------------------------------------
    def _check_common(self, sections: dict[str, str], use_template: bool) -> tuple[list[str], str]:
        """Section-completeness + banned-phrase checks shared by both profiles."""
        warnings: list[str] = []
        blob = " ".join(sections.values()).lower()
        if use_template:
            for sec in T.SECTIONS:
                if not sections.get(sec, "").strip():
                    warnings.append(f"missing or empty section: '{sec}'")
        elif not blob.strip():
            warnings.append("empty summary")
        for phrase in T.BANNED_PHRASES:
            if phrase in blob:
                warnings.append(f"overclaiming/diagnostic phrase detected: '{phrase.strip()}'")
        return warnings, blob

    def _validate_single(self, sections, profile: RiskProfile, use_template: bool) -> list[str]:
        warnings, blob = self._check_common(sections, use_template)
        pct = str(profile.probability_pct)
        text = sections.get("What we found", "") if use_template else " ".join(sections.values())
        if pct not in text:
            warnings.append("probability not faithfully stated")
        if profile.confidence_label == "lower" and "caution" not in blob and "less certain" not in blob:
            warnings.append("low confidence not reflected in summary")
        return warnings

    def _validate_health(self, sections, profile: HealthRiskProfile, use_template: bool) -> list[str]:
        warnings, blob = self._check_common(sections, use_template)
        top = _top_forward_risk(profile)
        if top is not None:
            text = sections.get("What we found", "") if use_template else " ".join(sections.values())
            if str(top.probability_pct) not in text:
                warnings.append("top forward-risk percentage not faithfully stated")
            if top.confidence_label == "lower" and "caution" not in blob and "less certain" not in blob:
                warnings.append("low confidence not reflected in summary")
        return warnings
