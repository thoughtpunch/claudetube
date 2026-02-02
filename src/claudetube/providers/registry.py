"""
claudetube.providers.registry - Provider discovery and instance management.

This module manages the registry of available providers, handling lazy loading,
caching, and provider discovery.

Functions:
    get_provider: Get a provider instance by name.
    list_available: List providers that are configured and ready.
    list_all: List all known provider names.
    clear_cache: Clear the provider instance cache.
    get_provider_info: Get provider info without fully initializing.

Example:
    >>> from claudetube.providers.registry import get_provider, list_available
    >>> provider = get_provider("whisper-local")
    >>> available = list_available()
    >>> print(available)
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudetube.providers.base import Provider

logger = logging.getLogger(__name__)


# Provider module mapping - maps canonical names to module paths
# Provider classes follow naming convention: {Name}Provider
# e.g., "whisper-local" -> WhisperLocalProvider, "openai" -> OpenaiProvider
PROVIDER_MODULES: dict[str, str] = {
    "whisper-local": "claudetube.providers.whisper_local",
    "openai": "claudetube.providers.openai",
    "anthropic": "claudetube.providers.anthropic",
    "google": "claudetube.providers.google",
    "deepgram": "claudetube.providers.deepgram",
    "assemblyai": "claudetube.providers.assemblyai",
    "voyage": "claudetube.providers.voyage",
    "local-embedder": "claudetube.providers.local_embedder",
    "ollama": "claudetube.providers.ollama",
    "claude-code": "claudetube.providers.claude_code",
    "litellm": "claudetube.providers.litellm",
}


# Aliases for convenience - allows users to use common names
# Maps alias -> canonical name
PROVIDER_ALIASES: dict[str, str] = {
    # Whisper aliases
    "whisper": "whisper-local",
    "faster-whisper": "whisper-local",
    # OpenAI model aliases
    "gpt-4o": "openai",
    "gpt4": "openai",
    "gpt-4": "openai",
    "gpt": "openai",
    "whisper-api": "openai",
    # Anthropic aliases
    "claude": "anthropic",
    "claude-3": "anthropic",
    "claude-sonnet": "anthropic",
    "claude-opus": "anthropic",
    "claude-haiku": "anthropic",
    # Google/Gemini aliases
    "gemini": "google",
    "gemini-2.0-flash": "google",
    "gemini-pro": "google",
    "gemini-flash": "google",
    # LiteLLM aliases
    "lite-llm": "litellm",
    "lite_llm": "litellm",
    # Embedding aliases
    "voyage-ai": "voyage",
    "voyage-3": "voyage",
    "local": "local-embedder",
}


# Cached provider instances - keyed by canonical name
# Only caches instances created without custom kwargs
_cache: dict[str, Provider] = {}


def _resolve_name(name: str) -> str:
    """Resolve provider aliases to canonical names.

    Args:
        name: Provider name or alias (case-insensitive).

    Returns:
        Canonical provider name.

    Example:
        >>> _resolve_name("gpt-4o")
        'openai'
        >>> _resolve_name("OpenAI")
        'openai'
    """
    normalized = name.lower().strip()
    return PROVIDER_ALIASES.get(normalized, normalized)


def _canonical_to_class_name(canonical: str) -> str:
    """Convert canonical provider name to class name.

    Convention: {Name}Provider where Name is title-cased with hyphens removed.

    Args:
        canonical: Canonical provider name (e.g., "whisper-local", "openai").

    Returns:
        Class name (e.g., "WhisperLocalProvider", "OpenaiProvider").

    Example:
        >>> _canonical_to_class_name("whisper-local")
        'WhisperLocalProvider'
        >>> _canonical_to_class_name("claude-code")
        'ClaudeCodeProvider'
    """
    parts = canonical.split("-")
    return "".join(part.title() for part in parts) + "Provider"


def get_provider(name: str, **kwargs) -> Provider:
    """Get a provider instance by name.

    Providers are lazily loaded - the module is only imported when first
    requested. Instances are cached for reuse unless custom kwargs are
    provided.

    Args:
        name: Provider name or alias (e.g., "openai", "gpt-4o", "whisper-local").
            Case-insensitive.
        **kwargs: Provider-specific configuration (e.g., model_size, api_key).
            If provided, instance is not cached.

    Returns:
        Provider instance ready for use.

    Raises:
        ValueError: If provider name is unknown.
        ImportError: If provider module fails to import (missing dependencies).

    Example:
        >>> provider = get_provider("whisper-local", model_size="small")
        >>> provider = get_provider("gpt-4o")  # Alias for openai
        >>> provider = get_provider("openai", model="gpt-4o-mini")
    """
    canonical = _resolve_name(name)

    # Check cache (only if no kwargs - kwargs might change config)
    cache_key = canonical if not kwargs else None
    if cache_key and cache_key in _cache:
        logger.debug(f"Returning cached provider: {canonical}")
        return _cache[cache_key]

    # Validate provider name
    if canonical not in PROVIDER_MODULES:
        available = ", ".join(sorted(PROVIDER_MODULES.keys()))
        aliases_for_display = ", ".join(
            f'"{alias}"' for alias in sorted(PROVIDER_ALIASES.keys())[:5]
        )
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Available providers: {available}. "
            f"Common aliases: {aliases_for_display}, etc."
        )

    # Lazy import the provider module
    module_path = PROVIDER_MODULES[canonical]
    try:
        module = import_module(module_path)
    except ImportError as e:
        # Provide helpful error message about missing dependencies
        dep_hints = {
            "openai": "pip install openai",
            "anthropic": "pip install anthropic",
            "google": "pip install google-generativeai",
            "deepgram": "pip install deepgram-sdk",
            "assemblyai": "pip install assemblyai",
            "voyage": "pip install voyageai",
            "local-embedder": "pip install sentence-transformers",
            "whisper-local": "pip install faster-whisper",
            "ollama": "pip install ollama",
            "litellm": "pip install litellm",
        }
        hint = dep_hints.get(canonical, "check the provider documentation")
        raise ImportError(
            f"Failed to import provider '{canonical}' from {module_path}: {e}. "
            f"You may need to install dependencies: {hint}"
        ) from e

    # Find the provider class
    class_name = _canonical_to_class_name(canonical)

    if not hasattr(module, class_name):
        # Fallback: look for any Provider subclass in the module
        from claudetube.providers.base import Provider as BaseProvider

        provider_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseProvider)
                and attr is not BaseProvider
                and not attr_name.startswith("_")
            ):
                provider_class = attr
                class_name = attr_name
                break

        if provider_class is None:
            raise ImportError(
                f"No Provider class found in {module_path}. "
                f"Expected class named '{_canonical_to_class_name(canonical)}' "
                f"or any subclass of Provider."
            )
    else:
        provider_class = getattr(module, class_name)

    # Instantiate the provider
    try:
        instance = provider_class(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate {class_name}: {e}. "
            f"Check that the kwargs match the provider's __init__ signature."
        ) from e

    # Cache if no custom kwargs
    if cache_key:
        _cache[cache_key] = instance
        logger.debug(f"Cached provider instance: {canonical}")

    logger.debug(f"Loaded provider: {canonical} ({class_name})")
    return instance


def list_available() -> list[str]:
    """List providers that are configured and ready to use.

    This function imports each provider module and checks if it's available
    (API key set, dependencies installed, etc.). This is relatively expensive
    so should be used sparingly.

    Returns:
        List of canonical provider names where is_available() returns True.
        Sorted alphabetically.

    Example:
        >>> available = list_available()
        >>> print(available)
        ['claude-code', 'whisper-local']
    """
    available = []

    for name in PROVIDER_MODULES:
        try:
            provider = get_provider(name)
            if provider.is_available():
                available.append(name)
                logger.debug(f"Provider '{name}' is available")
            else:
                logger.debug(f"Provider '{name}' not available (is_available=False)")
        except ImportError as e:
            logger.debug(f"Provider '{name}' not available (import error): {e}")
        except Exception as e:
            logger.debug(f"Provider '{name}' not available (error): {e}")

    return sorted(available)


def list_all() -> list[str]:
    """List all known provider names.

    Returns all registered provider names regardless of availability.
    Does not import provider modules or check availability.

    Returns:
        List of all canonical provider names, sorted alphabetically.

    Example:
        >>> all_providers = list_all()
        >>> print(all_providers)
        ['anthropic', 'assemblyai', 'claude-code', 'deepgram', ...]
    """
    return sorted(PROVIDER_MODULES.keys())


def clear_cache() -> None:
    """Clear the provider instance cache.

    Useful for testing or when configuration changes require fresh instances.
    After clearing, the next get_provider() call will create new instances.

    Example:
        >>> clear_cache()
        >>> provider = get_provider("openai")  # Creates new instance
    """
    _cache.clear()
    logger.debug("Provider cache cleared")


def get_provider_info(name: str) -> dict:
    """Get provider info without fully initializing the provider.

    Returns basic capability information from PROVIDER_INFO without
    importing the provider module or checking availability.

    Args:
        name: Provider name or alias.

    Returns:
        Dictionary with provider metadata:
        - name: Canonical provider name
        - capabilities: List of capability names (e.g., ["TRANSCRIBE", "VISION"])
        - supports_structured_output: Whether provider supports JSON schemas

    Example:
        >>> info = get_provider_info("gpt-4o")
        >>> print(info)
        {'name': 'openai', 'capabilities': ['TRANSCRIBE', 'VISION', 'REASON'], ...}
    """
    from claudetube.providers.capabilities import PROVIDER_INFO

    canonical = _resolve_name(name)

    if canonical in PROVIDER_INFO:
        info = PROVIDER_INFO[canonical]
        return {
            "name": info.name,
            "capabilities": [c.name for c in info.capabilities],
            "supports_structured_output": info.supports_structured_output,
            "supports_streaming": info.supports_streaming,
            "supports_diarization": info.supports_diarization,
            "supports_translation": info.supports_translation,
        }

    # Provider not in PROVIDER_INFO - return minimal info
    return {
        "name": canonical,
        "capabilities": [],
        "supports_structured_output": False,
        "supports_streaming": False,
        "supports_diarization": False,
        "supports_translation": False,
    }


def get_aliases() -> dict[str, str]:
    """Get the alias mapping.

    Returns:
        Dictionary mapping aliases to canonical provider names.

    Example:
        >>> aliases = get_aliases()
        >>> aliases["gpt-4o"]
        'openai'
    """
    return PROVIDER_ALIASES.copy()


def get_canonical_name(name: str) -> str:
    """Get the canonical name for a provider.

    Args:
        name: Provider name or alias.

    Returns:
        Canonical provider name.

    Raises:
        ValueError: If name doesn't map to a known provider.

    Example:
        >>> get_canonical_name("gpt-4o")
        'openai'
        >>> get_canonical_name("whisper")
        'whisper-local'
    """
    canonical = _resolve_name(name)
    if canonical not in PROVIDER_MODULES:
        available = ", ".join(sorted(PROVIDER_MODULES.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return canonical


__all__ = [
    # Primary API
    "get_provider",
    "list_available",
    "list_all",
    "clear_cache",
    "get_provider_info",
    # Utilities
    "get_aliases",
    "get_canonical_name",
    # Constants (for advanced use)
    "PROVIDER_MODULES",
    "PROVIDER_ALIASES",
]
