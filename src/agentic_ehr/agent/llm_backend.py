"""LLM summary backend — the core of the system.

Turns a structured :class:`RiskProfile` into the five-section patient summary
using a hosted LLM. Three providers are supported behind one interface:

* ``gemini``  — Google Gemini (the **default**; best free model). Needs
  ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and the ``google-genai`` SDK.
* ``claude``  — Anthropic Claude. Needs ``ANTHROPIC_API_KEY`` and ``anthropic``.
* ``openai``  — OpenAI GPT. Needs ``OPENAI_API_KEY`` and ``openai``.

Common design (per LLM best practice):

* **Structured outputs**: every provider is constrained to emit a strict JSON
  object with exactly the five sections, so we never parse free-form text.
* **One system prompt**: a single, stable, non-diagnostic safety prompt is shared
  across providers; the per-patient ``RiskProfile`` is the volatile user turn.
* **No fallback**: if the SDK or key is missing, or the API call fails, the
  backend raises :class:`SummaryBackendError` — the caller surfaces it.
"""
from __future__ import annotations

import json
import os

from ..explain.risk_profile import RiskProfile
from ..logging_utils import get_logger
from . import templates as T

logger = get_logger(__name__)

# The five required sections, as a strict JSON schema for structured outputs.
# Anthropic and OpenAI accept the full Draft-07 subset (incl. additionalProperties).
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {section: {"type": "string"} for section in T.SECTIONS},
    "required": list(T.SECTIONS),
    "additionalProperties": False,
}

# Gemini's response_schema uses the OpenAPI subset: no additionalProperties,
# ordering expressed via propertyOrdering.
_GEMINI_SCHEMA = {
    "type": "object",
    "properties": {section: {"type": "string"} for section in T.SECTIONS},
    "required": list(T.SECTIONS),
    "propertyOrdering": list(T.SECTIONS),
}

_SYSTEM_PROMPT = f"""You are a careful health-information assistant. You translate a
machine-learning model's risk estimate into a patient-friendly summary.

Hard rules — these are non-negotiable safety constraints:
- You are NOT a doctor. Never diagnose, never claim certainty, never tell the
  patient they "have", "will get", or are "going to" develop any disease.
- Use ONLY the numbers and factors in the structured input. Never invent
  symptoms, lab values, diagnoses, medications, or treatments.
- Reflect the model's stated uncertainty honestly. When the confidence label is
  "lower", explicitly tell the reader to treat the estimate with extra caution.
- Use plain, calm, non-alarming language a non-expert can read.
- Recommend discussing the results with a qualified clinician.

Content requirements per section:
- "What we found": state the estimated chance and include the exact whole-number
  percentage from the input (e.g. "about 42%"). Make clear it is a statistical
  estimate, not a prediction of what will happen to this person.
- "What may be contributing": describe the listed contributing factors in plain
  language as associations the model noticed, not proven causes.
- "What this means": give calm, balanced context; reflect the confidence level.
- "What to do next": non-prescriptive, educational next steps to discuss with a
  clinician. Never give treatment instructions or tell them to start/stop a drug.
- "When to seek care urgently": general, widely-recognised warning signs only —
  not tailored to this individual's estimate.

Avoid these phrasings entirely: "you have", "you will", "you are going to",
"diagnosed with", "is certain", "guaranteed", "you must take", "stop taking",
"no need to see".

Return exactly the five sections defined by the output schema. Each value is a
short, readable paragraph (the urgent-care section may use a short list)."""


class SummaryBackendError(RuntimeError):
    """Raised when the LLM backend cannot be constructed or invoked."""


def _user_prompt(profile: RiskProfile) -> str:
    payload = json.dumps(profile.to_dict(), indent=2, default=str)
    return (
        "Here is the structured risk profile. Produce the five-section summary, "
        "faithful to these numbers and factors only:\n\n" + payload
    )


def _parse_sections(text: str) -> dict[str, str]:
    if not text or not text.strip():
        raise SummaryBackendError("Model returned no text content.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SummaryBackendError(f"Model did not return valid JSON: {exc}") from exc
    return {section: str(data.get(section, "")).strip() for section in T.SECTIONS}


class _BaseLLMBackend:
    """Shared scaffolding. Subclasses set ``provider``/``default_model`` and
    implement :meth:`_make_client` and :meth:`generate`."""

    name = "llm"
    provider = "base"
    default_model = ""

    def __init__(self, model: str | None = None, max_tokens: int = 4000):
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self._client = self._make_client()

    def _make_client(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def generate(self, profile: RiskProfile) -> dict[str, str]:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Gemini (default)                                                            #
# --------------------------------------------------------------------------- #
class GeminiBackend(_BaseLLMBackend):
    provider = "gemini"
    default_model = "gemini-2.5-pro"  # best free model on Google AI Studio

    def _make_client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise SummaryBackendError(
                "The 'google-genai' package is required for the Gemini summary agent. "
                "Install it (`pip install google-genai`) or choose another provider."
            ) from exc
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SummaryBackendError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. The Gemini summary "
                "agent cannot run. Set the key or switch agent.llm.provider."
            )
        return genai.Client(api_key=api_key)

    def generate(self, profile: RiskProfile) -> dict[str, str]:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=_user_prompt(profile),
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                    temperature=0.3,
                    response_mime_type="application/json",
                    response_schema=_GEMINI_SCHEMA,
                ),
            )
        except Exception as exc:  # google.genai raises various error types
            raise SummaryBackendError(f"Gemini API call failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "LLM summary usage (gemini): prompt=%s output=%s",
                getattr(usage, "prompt_token_count", "?"),
                getattr(usage, "candidates_token_count", "?"),
            )
        return _parse_sections(getattr(response, "text", "") or "")


# --------------------------------------------------------------------------- #
# Anthropic Claude                                                            #
# --------------------------------------------------------------------------- #
class AnthropicBackend(_BaseLLMBackend):
    provider = "claude"
    default_model = "claude-opus-4-8"

    def _make_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise SummaryBackendError(
                "The 'anthropic' package is required for the Claude summary agent. "
                "Install it (`pip install anthropic`) or choose another provider."
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SummaryBackendError(
                "ANTHROPIC_API_KEY is not set. The Claude summary agent cannot run. "
                "Set the key or switch agent.llm.provider."
            )
        return anthropic.Anthropic()

    def generate(self, profile: RiskProfile) -> dict[str, str]:
        import anthropic

        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": _user_prompt(profile)}],
            )
        except anthropic.APIError as exc:
            raise SummaryBackendError(f"Anthropic API call failed: {exc}") from exc

        if getattr(message, "stop_reason", None) == "refusal":
            raise SummaryBackendError("The model refused to generate this summary.")

        usage = getattr(message, "usage", None)
        if usage is not None:
            logger.info(
                "LLM summary usage (claude): input=%s cache_read=%s output=%s",
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "cache_read_input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
            )
        text = next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")
        return _parse_sections(text)


# --------------------------------------------------------------------------- #
# OpenAI GPT                                                                  #
# --------------------------------------------------------------------------- #
class OpenAIBackend(_BaseLLMBackend):
    provider = "openai"
    default_model = "gpt-4o"

    def _make_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SummaryBackendError(
                "The 'openai' package is required for the GPT summary agent. "
                "Install it (`pip install openai`) or choose another provider."
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise SummaryBackendError(
                "OPENAI_API_KEY is not set. The GPT summary agent cannot run. "
                "Set the key or switch agent.llm.provider."
            )
        return OpenAI()

    def generate(self, profile: RiskProfile) -> dict[str, str]:
        from openai import OpenAIError

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(profile)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "patient_summary",
                        "schema": _OUTPUT_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except OpenAIError as exc:
            raise SummaryBackendError(f"OpenAI API call failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "LLM summary usage (openai): prompt=%s completion=%s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
            )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise SummaryBackendError("The model refused to generate this summary.")
        return _parse_sections(choice.message.content or "")


_PROVIDERS = {
    "gemini": GeminiBackend,
    "google": GeminiBackend,
    "claude": AnthropicBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "gpt": OpenAIBackend,
}


def make_llm_backend(
    provider: str = "gemini", model: str | None = None, max_tokens: int = 4000
) -> _BaseLLMBackend:
    """Construct an LLM backend for ``provider`` (gemini | claude | openai).

    ``model=None`` selects the provider's best default. Raises
    :class:`SummaryBackendError` for an unknown provider or a missing SDK/key.
    """
    key = (provider or "gemini").strip().lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        choices = sorted({"gemini", "claude", "openai"})
        raise SummaryBackendError(
            f"Unknown LLM provider: {provider!r}. Expected one of {choices}."
        )
    return cls(model=model, max_tokens=max_tokens)
