"""LLM baseline: feed raw EHR values straight to the LLM, no prediction model.

This is the ablation control for the agentic pipeline. It uses the SAME LLM backend and
the SAME output format (five-section template or free narrative) as :class:`SummaryAgent`,
but its input is the patient's raw recorded clinical values with NO model predictions, risk
scores, or attributions. Comparing its output against the full pipeline isolates the value
of the specialised prediction + attribution agents.

Unlike ``SummaryAgent``, guardrail violations are RECORDED, not raised: the baseline is
expected to overclaim more often, and that rate is itself a comparison signal, so we keep
the sample rather than abort the run.
"""
from __future__ import annotations

from ..logging_utils import get_logger
from . import templates as T
from .summary_agent import PatientSummary

logger = get_logger(__name__)


def _validate(sections: dict[str, str], use_template: bool) -> list[str]:
    """Same safety floor as the agent (sections present + banned phrases), minus the
    probability/confidence checks that don't apply without a model."""
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
    return warnings


class LLMBaselineAgent:
    """Raw-EHR-to-summary baseline (no prediction model, no attribution)."""

    def __init__(self, backend):
        if backend is None:
            raise ValueError("LLMBaselineAgent requires an explicit backend.")
        self.backend = backend

    @classmethod
    def from_config(cls, cfg) -> "LLMBaselineAgent":
        from .llm_backend import make_llm_backend
        backend = make_llm_backend(
            provider=cfg.get("agent.llm.provider", "gemini"),
            model=cfg.get("agent.llm.model", None),
            max_tokens=cfg.get("agent.llm.max_tokens", 4000),
        )
        return cls(backend)

    def summarize(self, raw_ehr: dict, use_template: bool = True) -> PatientSummary:
        """Generate a summary directly from a patient's raw EHR values.

        ``raw_ehr`` is a plain dict of recorded clinical values (see
        ``MultiTaskInferenceService.raw_ehr_payload_for``). Guardrail warnings are attached
        to the returned summary rather than raised.
        """
        sections = self.backend.run_sections(
            T.BASELINE_SYSTEM_PROMPT, T.BASELINE_SYSTEM_PROMPT_FREE, raw_ehr, use_template,
        )
        warnings = _validate(sections, use_template)
        if warnings:
            logger.warning("Baseline guardrail warnings (recorded, not raised): %s", warnings)
        return PatientSummary(
            sections=sections,
            disclaimer=T.DISCLAIMER,
            backend=self.backend.name + "-baseline",
            risk_tier="n/a",
            probability_pct=0,
            mode="template" if use_template else "free",
            guardrail_warnings=warnings,
        )
