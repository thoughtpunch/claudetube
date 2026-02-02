"""
claudetube.providers.litellm.client - LiteLLM provider implementation.

Uses LiteLLM's unified API for reasoning across 100+ LLM providers.
LiteLLM translates OpenAI-format calls to provider-native APIs, so any
model accessible through LiteLLM can be used for reasoning tasks.

Model format examples:
    - "gpt-4o" (OpenAI)
    - "anthropic/claude-sonnet-4-20250514" (Anthropic)
    - "bedrock/anthropic.claude-3-sonnet" (AWS Bedrock)
    - "azure/my-deployment" (Azure OpenAI)
    - "ollama/llama3.2" (Ollama)
    - "together_ai/meta-llama/Llama-3-70b" (Together AI)

See https://docs.litellm.ai/docs/providers for the full list.

Example:
    >>> provider = LitellmProvider(model="anthropic/claude-sonnet-4-20250514")
    >>> result = await provider.reason(
    ...     [{"role": "user", "content": "Summarize this transcript..."}]
    ... )
    >>> print(result)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from claudetube.providers.base import Provider, Reasoner
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"


def _get_schema_json(schema: type) -> str:
    """Extract JSON schema from a Pydantic model or type.

    Args:
        schema: Pydantic BaseModel subclass or other type.

    Returns:
        JSON string of the schema.
    """
    if hasattr(schema, "model_json_schema"):
        return json.dumps(schema.model_json_schema(), indent=2)
    return str(schema)


class LitellmProvider(Provider, Reasoner):
    """LiteLLM provider for generic reasoning across 100+ LLM providers.

    Uses LiteLLM's unified API to route requests to any supported backend.
    The model string determines which provider and model is used (e.g.,
    "anthropic/claude-sonnet-4-20250514", "gpt-4o", "bedrock/anthropic.claude-3-sonnet").

    Args:
        model: LiteLLM model string. Defaults to "gpt-4o".
        api_key: API key for the underlying provider. If not set, LiteLLM
            will look for the appropriate env var (e.g., OPENAI_API_KEY,
            ANTHROPIC_API_KEY) based on the model string.
        api_base: Custom API base URL (for proxies or custom endpoints).
        max_tokens: Default max tokens for responses.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._max_tokens = max_tokens

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["litellm"]

    def is_available(self) -> bool:
        """Check if the LiteLLM package is installed.

        LiteLLM handles API key resolution internally based on the model
        string, so we only check for the package being importable.
        """
        try:
            import litellm  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("LITELLM_API_KEY")

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response using LiteLLM's unified API.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            schema: Optional Pydantic model for structured output. Uses
                prompt-based JSON requesting since not all LiteLLM backends
                support native JSON mode.
            **kwargs: Additional options passed to litellm.acompletion:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.
                temperature (float): Sampling temperature.
                Any other litellm.acompletion parameter.

        Returns:
            str if no schema, dict if schema provided and JSON parseable.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm not installed. Install with: pip install litellm"
            ) from e

        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        # Build messages, appending schema request if needed
        api_messages = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in messages
        ]

        if schema and api_messages:
            schema_json = _get_schema_json(schema)
            schema_suffix = (
                "\n\nRespond with JSON matching this schema:\n"
                f"```json\n{schema_json}\n```"
            )
            # Append to the last user message
            for i in range(len(api_messages) - 1, -1, -1):
                if api_messages[i]["role"] == "user":
                    api_messages[i] = {
                        "role": "user",
                        "content": api_messages[i]["content"] + schema_suffix,
                    }
                    break

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Pass API key and base if configured
        api_key = self._resolve_api_key()
        if api_key:
            api_kwargs["api_key"] = api_key
        if self._api_base:
            api_kwargs["api_base"] = self._api_base

        response = await litellm.acompletion(**api_kwargs)
        text = response.choices[0].message.content

        if schema:
            return self._try_parse_json(text)

        return text

    @staticmethod
    def _try_parse_json(text: str) -> str | dict:
        """Try to parse text as JSON, return raw text on failure.

        Handles cases where the model wraps JSON in markdown code blocks.

        Args:
            text: Response text that may contain JSON.

        Returns:
            Parsed dict if valid JSON, original text otherwise.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            if len(lines) >= 3 and lines[-1].strip() == "```":
                inner = "\n".join(lines[1:-1])
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass

        return text
