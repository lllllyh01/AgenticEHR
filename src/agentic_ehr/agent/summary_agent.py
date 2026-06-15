"""The SummaryAgent: drives an LLM backend and enforces safety guardrails.

The AI (LLM) summary agent is the core of the system — and the only summary
generator. There is **no fallback**: if the backend cannot run (missing SDK /
API key, or an API error) or its output fails the safety guardrails, the agent
RAISES so the failure is surfaced rather than masked. The backend is
provider-agnostic (Gemini / Claude / GPT); see :mod:`agent.llm_backend`.

The constructor accepts any object implementing the backend contract
(``name`` + ``generate(profile)``), which keeps the agent decoupled from the
provider and lets callers inject an alternative backend (e.g. a test double).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..explain.risk_profile import RiskProfile
from ..logging_utils import get_logger
from . import templates as T

logger = get_logger(__name__)


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

    def summarize(self, profile: RiskProfile, use_template: bool = True) -> PatientSummary:
        """Generate a summary. Raises on backend failure or guardrail violation.

        ``use_template=True`` produces the five fixed sections (structured
        output); ``use_template=False`` produces one free-form narrative. The
        safety guardrails apply in both modes. No fallback: a failure here is
        surfaced to the caller, never replaced by a silently-substituted summary.
        """
        sections = self.backend.generate(profile, use_template=use_template)  # may raise
        warnings = self._validate(sections, profile, use_template)
        if warnings:
            logger.error("Guardrail violations from %s backend: %s", self.backend.name, warnings)
            raise SummaryGuardrailError(warnings)

        return PatientSummary(
            sections=sections,
            disclaimer=T.DISCLAIMER,
            backend=self.backend.name,
            risk_tier=profile.risk_tier,
            probability_pct=profile.probability_pct,
            mode="template" if use_template else "free",
            guardrail_warnings=[],
        )

    # ----- guardrails --------------------------------------------------------
    def _validate(
        self, sections: dict[str, str], profile: RiskProfile, use_template: bool = True
    ) -> list[str]:
        warnings: list[str] = []
        blob = " ".join(sections.values()).lower()
        pct = str(profile.probability_pct)

        if use_template:
            for sec in T.SECTIONS:
                if not sections.get(sec, "").strip():
                    warnings.append(f"missing or empty section: '{sec}'")
            # Faithfulness: the stated percentage must appear in "What we found".
            if pct not in sections.get("What we found", ""):
                warnings.append("probability not faithfully stated in 'What we found'")
        else:
            if not blob.strip():
                warnings.append("empty summary")
            # Faithfulness: the stated percentage must appear somewhere.
            if pct not in " ".join(sections.values()):
                warnings.append("probability not faithfully stated in summary")

        for phrase in T.BANNED_PHRASES:
            if phrase in blob:
                warnings.append(f"overclaiming/diagnostic phrase detected: '{phrase.strip()}'")

        # Uncertainty must be acknowledged when confidence is lower.
        if profile.confidence_label == "lower" and "caution" not in blob and "less certain" not in blob:
            warnings.append("low confidence not reflected in summary")

        return warnings
