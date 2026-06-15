"""Pragmatic, automatic checks on summary quality.

These are heuristic proxies, not a substitute for clinical review. They check:
  * factual consistency  - the stated risk % matches the RiskProfile
  * no unsupported claims - no banned diagnostic/overclaiming phrases; named
                            contributors actually come from the profile
  * uncertainty faithful  - low confidence is acknowledged in the text
  * clarity               - readability proxy (sentence length, all sections present)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..agent.summary_agent import PatientSummary
from ..agent import templates as T
from ..explain.risk_profile import RiskProfile


@dataclass
class SummaryQuality:
    factual_consistency: bool
    no_unsupported_claims: bool
    uncertainty_faithful: bool
    all_sections_present: bool
    clarity_score: float           # 0..1, higher = clearer
    issues: list[str]

    @property
    def passed(self) -> bool:
        return (
            self.factual_consistency
            and self.no_unsupported_claims
            and self.uncertainty_faithful
            and self.all_sections_present
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def evaluate_summary(summary: PatientSummary, profile: RiskProfile) -> SummaryQuality:
    sections = summary.sections
    blob = " ".join(sections.values())
    blob_lower = blob.lower()
    issues: list[str] = []

    # 1) Factual consistency: the stated percentage matches the profile.
    pct = profile.probability_pct
    stated = _extract_percentages(blob)
    factual = pct in stated
    if not factual:
        issues.append(f"stated percentages {stated} do not include profile value {pct}%")

    # 2) No unsupported claims: banned phrases + contributor grounding.
    banned_hit = [p.strip() for p in T.BANNED_PHRASES if p in blob_lower]
    grounded = _contributors_grounded(sections.get("What may be contributing", ""), profile)
    no_unsupported = not banned_hit and grounded
    if banned_hit:
        issues.append(f"banned/overclaiming phrases: {banned_hit}")
    if not grounded:
        issues.append("contributing factors not grounded in the risk profile")

    # 3) Uncertainty faithfulness.
    if profile.confidence_label == "lower":
        unc_ok = ("caution" in blob_lower) or ("less certain" in blob_lower) or ("uncertain" in blob_lower)
    else:
        unc_ok = True
    if not unc_ok:
        issues.append("low model confidence not reflected in the summary")

    # 4) All sections present and non-empty.
    present = all(sections.get(s, "").strip() for s in T.SECTIONS)
    if not present:
        issues.append("one or more required sections missing/empty")

    clarity = _clarity_score(sections)

    return SummaryQuality(
        factual_consistency=factual,
        no_unsupported_claims=no_unsupported,
        uncertainty_faithful=unc_ok,
        all_sections_present=present,
        clarity_score=clarity,
        issues=issues,
    )


def _extract_percentages(text: str) -> set[int]:
    return {int(m) for m in re.findall(r"(\d{1,3})\s*%", text)}


def _contributors_grounded(text: str, profile: RiskProfile) -> bool:
    """If the section names contributing factors, at least one should match a
    contributor in the profile. If the profile has contributors but none are
    mentioned, that's an ungrounded/empty explanation."""
    if not profile.contributors:
        return True  # nothing to ground against
    text_lower = text.lower()
    concept_words = []
    for c in profile.contributors:
        concept_words += _keywords(c.patient_phrase) + _keywords(c.concept)
    return any(w in text_lower for w in concept_words if len(w) > 3)


def _keywords(phrase: str) -> list[str]:
    return [w.lower().strip(",.()") for w in phrase.split()]


def _clarity_score(sections: dict[str, str]) -> float:
    """Readability proxy: penalise very long sentences and missing sections."""
    text = " ".join(sections.values())
    sentences = [s for s in re.split(r"[.!?\n]", text) if s.strip()]
    if not sentences:
        return 0.0
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    # Ideal ~ 12-18 words/sentence; score decays outside that.
    length_score = max(0.0, 1.0 - abs(avg_words - 15) / 25.0)
    completeness = sum(1 for s in T.SECTIONS if sections.get(s, "").strip()) / len(T.SECTIONS)
    return round(0.5 * length_score + 0.5 * completeness, 3)
