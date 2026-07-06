"""LLM summary backend — the core of the system.

Turns a structured :class:`RiskProfile` into a patient-friendly summary using a
hosted LLM. Three providers are supported behind one interface:

* ``gemini``  — Google Gemini (the **default**; best free model). Needs
  ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and the ``google-genai`` SDK.
* ``claude``  — Anthropic Claude. Needs ``ANTHROPIC_API_KEY`` and ``anthropic``.
* ``openai``  — OpenAI GPT. Needs ``OPENAI_API_KEY`` and ``openai``.

Two output modes, both driven by the same non-diagnostic safety prompt:

* **template** (default): the model is constrained to **structured outputs** — a
  strict JSON object with exactly the five fixed sections, so we never parse
  free-form text. The section set lives in :mod:`agent.templates`.
* **free**: the model writes one cohesive, free-form patient-friendly summary
  (no fixed sections, no JSON schema). The same safety rules still apply.

No fallback: if the SDK or key is missing, or the API call fails, the backend
raises :class:`SummaryBackendError` — the caller surfaces it.
"""
from __future__ import annotations

import json
import os

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

class SummaryBackendError(RuntimeError):
    """Raised when the LLM backend cannot be constructed or invoked."""


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
    implement :meth:`_make_client`, :meth:`_complete`, and :meth:`_schema`."""

    name = "llm"
    provider = "base"
    default_model = ""

    def __init__(self, model: str | None = None, max_tokens: int = 4000):
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self._client = self._make_client()

    def _make_client(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _schema(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _complete(self, system: str, user: str, schema) -> str:  # pragma: no cover - abstract
        """Run one completion. ``schema=None`` requests free-form text;
        otherwise JSON constrained to ``schema``. Returns the raw text."""
        raise NotImplementedError

    def run_sections(
        self, system_template: str, system_free: str, payload: dict, use_template: bool = True
    ) -> dict[str, str]:
        """Generic section generator. Callers supply the system prompts and a
        JSON-serialisable ``payload`` (the structured input). Returns either the
        five fixed sections (template) or ``{"Summary": text}`` (free)."""
        user = (
            "Here is the structured input. Produce the summary, faithful to these "
            "numbers and factors only:\n\n" + json.dumps(payload, indent=2, default=str)
        )
        if use_template:
            return _parse_sections(self._complete(system_template, user, self._schema()))
        text = self._complete(system_free, user, None)
        if not text or not text.strip():
            raise SummaryBackendError("Model returned no text content.")
        return {"Summary": text.strip()}


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

    def _schema(self):
        return _GEMINI_SCHEMA

    def _complete(self, system: str, user: str, schema) -> str:
        from google.genai import types

        config_kwargs = dict(
            system_instruction=system,
            max_output_tokens=self.max_tokens,
            temperature=0.3,
        )
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(**config_kwargs),
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
        return getattr(response, "text", "") or ""


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

    def _schema(self):
        return _OUTPUT_SCHEMA

    def _complete(self, system: str, user: str, schema) -> str:
        import anthropic

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            message = self._client.messages.create(**kwargs)
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
        return next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")


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

    def _schema(self):
        return _OUTPUT_SCHEMA

    def _complete(self, system: str, user: str, schema) -> str:
        from openai import OpenAIError

        kwargs = dict(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "patient_summary", "schema": schema, "strict": True},
            }
        try:
            response = self._client.chat.completions.create(**kwargs)
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
        return choice.message.content or ""


_PROVIDERS = {
    "gemini": GeminiBackend,
    "claude": AnthropicBackend,
    "openai": OpenAIBackend,
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
