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
    guardrail_warnings: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = []
        for sec in T.SECTIONS:
            parts.append(f"## {sec}\n{self.sections.get(sec, '').strip()}")
        parts.append(f"---\n_{self.disclaimer}_")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "sections": self.sections,
            "disclaimer": self.disclaimer,
            "backend": self.backend,
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

    def summarize(self, profile: RiskProfile) -> PatientSummary:
        """Generate a summary. Raises on backend failure or guardrail violation.

        No fallback: a failure here is surfaced to the caller, never replaced by
        a silently-substituted summary.
        """
        sections = self.backend.generate(profile)  # may raise SummaryBackendError
        warnings = self._validate(sections, profile)
        if warnings:
            logger.error("Guardrail violations from %s backend: %s", self.backend.name, warnings)
            raise SummaryGuardrailError(warnings)

        return PatientSummary(
            sections=sections,
            disclaimer=T.DISCLAIMER,
            backend=self.backend.name,
            risk_tier=profile.risk_tier,
            probability_pct=profile.probability_pct,
            guardrail_warnings=[],
        )

    # ----- guardrails --------------------------------------------------------
    def _validate(self, sections: dict[str, str], profile: RiskProfile) -> list[str]:
        warnings: list[str] = []

        for sec in T.SECTIONS:
            if not sections.get(sec, "").strip():
                warnings.append(f"missing or empty section: '{sec}'")

        blob = " ".join(sections.values()).lower()
        for phrase in T.BANNED_PHRASES:
            if phrase in blob:
                warnings.append(f"overclaiming/diagnostic phrase detected: '{phrase.strip()}'")

        # Faithfulness: the stated percentage must match the profile.
        pct = str(profile.probability_pct)
        found_section = sections.get("What we found", "")
        if pct not in found_section:
            warnings.append("probability not faithfully stated in 'What we found'")

        # Uncertainty must be acknowledged when confidence is lower.
        if profile.confidence_label == "lower" and "caution" not in blob and "less certain" not in blob:
            warnings.append("low confidence not reflected in summary")

        return warnings
