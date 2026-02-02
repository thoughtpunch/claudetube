"""
claudetube.providers.config - Provider configuration loading.

Loads provider configuration from YAML config files with support for:
- ${ENV_VAR} interpolation for API keys
- Provider-specific settings (api_key, model, extras)
- Capability preferences (which provider for transcription, vision, etc.)
- Fallback chains for graceful degradation
- Backward compatibility with existing CLAUDETUBE_* env vars

Config is loaded from the same YAML files as the base config:
1. Project config (.claudetube/config.yaml)
2. User config (~/.config/claudetube/config.yaml)

YAML structure:
    providers:
      openai:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4o
      anthropic:
        api_key: ${ANTHROPIC_API_KEY}
      preferences:
        transcription: whisper-local
        vision: claude-code
        reasoning: claude-code
      fallbacks:
        vision: [anthropic, openai, claude-code]
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Pattern for ${ENV_VAR} interpolation
ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# Known provider names that have dedicated config fields (API-key providers)
_KNOWN_PROVIDERS = frozenset(
    {"openai", "anthropic", "google", "deepgram", "assemblyai", "voyage", "litellm"}
)

# All valid provider names (including local/built-in)
_ALL_PROVIDERS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "deepgram",
        "assemblyai",
        "voyage",
        "whisper-local",
        "claude-code",
        "ollama",
        "local-embedder",
        "litellm",
    }
)

# Valid whisper model sizes
_VALID_WHISPER_MODELS = frozenset({"tiny", "base", "small", "medium", "large"})

# Valid top-level keys in the providers section
_VALID_PROVIDERS_KEYS = _KNOWN_PROVIDERS | frozenset(
    {"local", "preferences", "fallbacks"}
)

# Maps capability name to the Capability enum member name
_CAPABILITY_FOR_PREFERENCE: dict[str, str] = {
    "transcription": "TRANSCRIBE",
    "vision": "VISION",
    "video": "VIDEO",
    "reasoning": "REASON",
    "embedding": "EMBED",
}

_VALID_PREFERENCE_KEYS = frozenset(
    {"transcription", "vision", "video", "reasoning", "embedding", "cost_preference"}
)

# Valid cost preference values
_VALID_COST_PREFERENCES = frozenset({"cost", "quality"})

_VALID_FALLBACK_KEYS = frozenset({"transcription", "vision", "reasoning"})

# Map from provider name to legacy env var for API key
_LEGACY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "assemblyai": "ASSEMBLYAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "litellm": "LITELLM_API_KEY",
}


@dataclass
class ProviderConfig:
    """Configuration for a single AI provider.

    Attributes:
        api_key: Resolved API key (after env var interpolation). None if not set.
        model: Default model name for this provider. None uses provider default.
        extra: Additional provider-specific settings.
    """

    api_key: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvidersConfig:
    """Complete providers configuration.

    Contains per-provider configs, local provider settings, capability
    preferences, and fallback chains.

    Attributes:
        providers: Per-provider configuration keyed by canonical name.
        whisper_local_model: Whisper model size for local transcription.
        ollama_model: Model name for Ollama.
        transcription_provider: Preferred transcription provider.
        vision_provider: Preferred vision provider.
        video_provider: Preferred video provider (only Gemini supports native).
        reasoning_provider: Preferred reasoning provider.
        embedding_provider: Preferred embedding provider.
        transcription_fallbacks: Fallback chain for transcription.
        vision_fallbacks: Fallback chain for vision.
        reasoning_fallbacks: Fallback chain for reasoning.
    """

    # Per-provider configs keyed by canonical name
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Local provider configs
    whisper_local_model: str = "small"
    ollama_model: str = "llava:13b"

    # Preferences (which provider to use for each capability)
    transcription_provider: str = "whisper-local"
    vision_provider: str = "claude-code"
    video_provider: str | None = None
    reasoning_provider: str = "claude-code"
    embedding_provider: str = "voyage"

    # Cost preference: "cost" prefers cheaper providers, "quality" uses config order
    cost_preference: str = "cost"

    # Fallback chains
    transcription_fallbacks: list[str] = field(
        default_factory=lambda: ["whisper-local"]
    )
    vision_fallbacks: list[str] = field(default_factory=lambda: ["claude-code"])
    reasoning_fallbacks: list[str] = field(default_factory=lambda: ["claude-code"])

    # Parallel fallback: try providers concurrently, return first success.
    # Keyed by capability name (transcription, vision, reasoning, embedding).
    parallel_fallback: dict[str, bool] = field(default_factory=dict)

    def get_provider_config(self, name: str) -> ProviderConfig:
        """Get config for a specific provider, creating default if needed.

        Args:
            name: Canonical provider name.

        Returns:
            ProviderConfig for the provider.
        """
        if name not in self.providers:
            self.providers[name] = ProviderConfig()
        return self.providers[name]


def _interpolate_env_vars(value: Any) -> Any:
    """Replace ${ENV_VAR} patterns with environment variable values.

    Recursively processes strings, dicts, and lists. Missing env vars
    produce a warning and are replaced with empty string.

    Args:
        value: Value to interpolate. Can be str, dict, list, or other.

    Returns:
        Value with all ${ENV_VAR} patterns resolved.
    """
    if isinstance(value, str):

        def _replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(
                    "Environment variable %s not set (referenced in provider config)",
                    var_name,
                )
                return ""
            return env_value

        return ENV_VAR_PATTERN.sub(_replace_match, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def _load_single_provider_config(data: dict) -> ProviderConfig:
    """Load a single provider's config from a dict.

    Performs env var interpolation on all values. Extracts api_key and model
    as dedicated fields, everything else goes into extra.

    Args:
        data: Raw config dict for one provider.

    Returns:
        ProviderConfig with resolved values.
    """
    resolved = _interpolate_env_vars(data)
    return ProviderConfig(
        api_key=resolved.get("api_key") or None,
        model=resolved.get("model"),
        extra={k: v for k, v in resolved.items() if k not in ("api_key", "model")},
    )


def _apply_legacy_env_vars(config: ProvidersConfig) -> None:
    """Apply legacy CLAUDETUBE_* env vars as fallbacks.

    For backward compatibility, if a provider has no api_key set in the
    YAML config but has a legacy env var set, use that.

    Also checks for direct provider env vars (OPENAI_API_KEY, etc.).

    Args:
        config: ProvidersConfig to update in-place.
    """
    for provider_name, env_var in _LEGACY_ENV_VARS.items():
        pc = config.get_provider_config(provider_name)
        if pc.api_key:
            continue

        # Check direct env var (e.g., OPENAI_API_KEY)
        direct_value = os.environ.get(env_var)
        if direct_value:
            pc.api_key = direct_value
            continue

        # Check CLAUDETUBE_ prefixed env var (e.g., CLAUDETUBE_OPENAI_API_KEY)
        prefixed_var = f"CLAUDETUBE_{env_var}"
        prefixed_value = os.environ.get(prefixed_var)
        if prefixed_value:
            pc.api_key = prefixed_value


@dataclass
class ConfigValidationResult:
    """Result of validating a providers config dict.

    Attributes:
        errors: Fatal issues that prevent correct operation.
        warnings: Non-fatal issues that may cause unexpected behavior.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors were found."""
        return len(self.errors) == 0


def _resolve_provider_name(name: str) -> str:
    """Resolve a provider alias to its canonical name.

    Uses the registry's alias mapping if available, otherwise returns
    the lowercased name.
    """
    try:
        from claudetube.providers.registry import PROVIDER_ALIASES

        normalized = name.lower().strip()
        return PROVIDER_ALIASES.get(normalized, normalized)
    except ImportError:
        return name.lower().strip()


def _check_provider_capability(
    provider_name: str, capability_name: str
) -> str | None:
    """Check if a provider supports a capability.

    Returns an error message if the provider doesn't support the capability,
    or None if it does (or if we can't check).
    """
    try:
        from claudetube.providers.capabilities import PROVIDER_INFO, Capability

        cap = getattr(Capability, capability_name, None)
        if cap is None:
            return None

        canonical = _resolve_provider_name(provider_name)
        info = PROVIDER_INFO.get(canonical)
        if info is None:
            return None

        if not info.can(cap):
            available_caps = ", ".join(
                c.name.lower() for c in info.capabilities
            )
            return (
                f"Provider '{provider_name}' does not support "
                f"{capability_name.lower()}. "
                f"Its capabilities: {available_caps or 'none'}"
            )
    except ImportError:
        pass
    return None


def validate_providers_config(
    config_dict: dict | None = None,
) -> ConfigValidationResult:
    """Validate a providers configuration dict.

    Checks for:
    - Unknown keys in the providers section
    - Invalid provider names in preferences and fallbacks
    - Capability mismatches (e.g., vision preference set to a transcription-only provider)
    - Invalid whisper model sizes
    - Invalid preference and fallback key names
    - Structural issues (wrong types)

    Args:
        config_dict: Parsed YAML config dict (the full config, not just providers section).

    Returns:
        ConfigValidationResult with errors and warnings.
    """
    result = ConfigValidationResult()

    if config_dict is None:
        return result

    if not isinstance(config_dict, dict):
        result.errors.append(
            "Config must be a YAML mapping (dict), "
            f"got {type(config_dict).__name__}"
        )
        return result

    providers_section = config_dict.get("providers")
    if providers_section is None:
        return result

    if not isinstance(providers_section, dict):
        result.errors.append(
            "The 'providers' section must be a mapping (dict), "
            f"got {type(providers_section).__name__}"
        )
        return result

    # Check for unknown top-level keys
    for key in providers_section:
        if key not in _VALID_PROVIDERS_KEYS:
            result.warnings.append(
                f"Unknown key '{key}' in providers section. "
                f"Valid keys: {', '.join(sorted(_VALID_PROVIDERS_KEYS))}"
            )

    # Validate provider configs (must be dicts)
    for provider_name in _KNOWN_PROVIDERS:
        if provider_name in providers_section:
            raw = providers_section[provider_name]
            if not isinstance(raw, dict):
                result.errors.append(
                    f"Config for provider '{provider_name}' must be a mapping (dict), "
                    f"got {type(raw).__name__}. "
                    f"Example: {provider_name}:\n"
                    f"           api_key: ${{YOUR_API_KEY}}\n"
                    f"           model: your-model"
                )

    # Validate local config
    local = providers_section.get("local")
    if local is not None:
        if not isinstance(local, dict):
            result.errors.append(
                "The 'local' section must be a mapping (dict), "
                f"got {type(local).__name__}"
            )
        else:
            # Validate whisper model
            whisper_model = local.get("whisper_model")
            if whisper_model is not None:
                model_str = str(whisper_model)
                if model_str not in _VALID_WHISPER_MODELS:
                    result.errors.append(
                        f"Invalid whisper_model '{model_str}'. "
                        f"Valid models: {', '.join(sorted(_VALID_WHISPER_MODELS))}"
                    )

            # Check for unknown local keys
            valid_local_keys = {"whisper_model", "ollama_model"}
            for key in local:
                if key not in valid_local_keys:
                    result.warnings.append(
                        f"Unknown key '{key}' in local section. "
                        f"Valid keys: {', '.join(sorted(valid_local_keys))}"
                    )

    # Validate preferences
    prefs = providers_section.get("preferences")
    if prefs is not None:
        if not isinstance(prefs, dict):
            result.errors.append(
                "The 'preferences' section must be a mapping (dict), "
                f"got {type(prefs).__name__}"
            )
        else:
            for key in prefs:
                if key not in _VALID_PREFERENCE_KEYS:
                    result.warnings.append(
                        f"Unknown preference '{key}'. "
                        f"Valid preferences: {', '.join(sorted(_VALID_PREFERENCE_KEYS))}"
                    )

            # Validate cost_preference
            cost_pref = prefs.get("cost_preference")
            if cost_pref is not None:
                cost_pref_str = str(cost_pref).lower()
                if cost_pref_str not in _VALID_COST_PREFERENCES:
                    result.errors.append(
                        f"Invalid cost_preference '{cost_pref}'. "
                        f"Valid values: {', '.join(sorted(_VALID_COST_PREFERENCES))}"
                    )

            for pref_key, cap_name in _CAPABILITY_FOR_PREFERENCE.items():
                if pref_key not in prefs:
                    continue
                provider_name = str(prefs[pref_key])
                canonical = _resolve_provider_name(provider_name)

                # Check if it's a known provider
                if canonical not in _ALL_PROVIDERS:
                    result.warnings.append(
                        f"Preference '{pref_key}' references unknown provider "
                        f"'{provider_name}'. "
                        f"Known providers: {', '.join(sorted(_ALL_PROVIDERS))}"
                    )
                    continue

                # Check capability match
                cap_err = _check_provider_capability(canonical, cap_name)
                if cap_err:
                    result.warnings.append(
                        f"Preference '{pref_key}': {cap_err}"
                    )

    # Validate fallbacks
    fallbacks = providers_section.get("fallbacks")
    if fallbacks is not None:
        if not isinstance(fallbacks, dict):
            result.errors.append(
                "The 'fallbacks' section must be a mapping (dict), "
                f"got {type(fallbacks).__name__}"
            )
        else:
            for key in fallbacks:
                if key not in _VALID_FALLBACK_KEYS:
                    result.warnings.append(
                        f"Unknown fallback key '{key}'. "
                        f"Valid fallback keys: {', '.join(sorted(_VALID_FALLBACK_KEYS))}"
                    )

            for fb_key, cap_name in _CAPABILITY_FOR_PREFERENCE.items():
                if fb_key not in fallbacks or fb_key not in _VALID_FALLBACK_KEYS:
                    continue
                chain = fallbacks[fb_key]
                if not isinstance(chain, list):
                    result.errors.append(
                        f"Fallback chain '{fb_key}' must be a list, "
                        f"got {type(chain).__name__}. "
                        f"Example: {fb_key}: [provider1, provider2]"
                    )
                    continue

                for provider_name in chain:
                    provider_str = str(provider_name)
                    canonical = _resolve_provider_name(provider_str)
                    if canonical not in _ALL_PROVIDERS:
                        result.warnings.append(
                            f"Fallback chain '{fb_key}' references unknown "
                            f"provider '{provider_str}'. "
                            f"Known providers: {', '.join(sorted(_ALL_PROVIDERS))}"
                        )
                    else:
                        cap_err = _check_provider_capability(
                            canonical, cap_name
                        )
                        if cap_err:
                            result.warnings.append(
                                f"Fallback chain '{fb_key}': {cap_err}"
                            )

    return result


def load_providers_config(config_dict: dict | None = None) -> ProvidersConfig:
    """Load providers configuration from a config dict.

    Processes the 'providers' section of the YAML config, resolving env vars
    and building the complete configuration. Falls back to defaults for any
    missing values.

    Args:
        config_dict: Parsed YAML config dict. If None, uses empty defaults.

    Returns:
        ProvidersConfig with all settings resolved.
    """
    if config_dict is None:
        config_dict = {}

    # Validate and log issues
    validation = validate_providers_config(config_dict)
    for error in validation.errors:
        logger.error("Config error: %s", error)
    for warning in validation.warnings:
        logger.warning("Config warning: %s", warning)

    providers_section = config_dict.get("providers", {})
    if not isinstance(providers_section, dict):
        providers_section = {}

    config = ProvidersConfig()

    # Load provider-specific configs
    for provider_name in _KNOWN_PROVIDERS:
        if provider_name in providers_section:
            raw = providers_section[provider_name]
            if isinstance(raw, dict):
                config.providers[provider_name] = _load_single_provider_config(raw)
            else:
                logger.warning(
                    "Invalid config for provider '%s' (expected dict)", provider_name
                )

    # Load local configs
    local = providers_section.get("local", {})
    if isinstance(local, dict):
        if "whisper_model" in local:
            config.whisper_local_model = str(local["whisper_model"])
        if "ollama_model" in local:
            config.ollama_model = str(local["ollama_model"])

    # Load preferences
    prefs = providers_section.get("preferences", {})
    if isinstance(prefs, dict):
        if "transcription" in prefs:
            config.transcription_provider = str(prefs["transcription"])
        if "vision" in prefs:
            config.vision_provider = str(prefs["vision"])
        if "video" in prefs:
            config.video_provider = str(prefs["video"])
        if "reasoning" in prefs:
            config.reasoning_provider = str(prefs["reasoning"])
        if "embedding" in prefs:
            config.embedding_provider = str(prefs["embedding"])
        if "cost_preference" in prefs:
            config.cost_preference = str(prefs["cost_preference"]).lower()

    # Load fallbacks
    fallbacks = providers_section.get("fallbacks", {})
    if isinstance(fallbacks, dict):
        if "transcription" in fallbacks and isinstance(
            fallbacks["transcription"], list
        ):
            config.transcription_fallbacks = [
                str(x) for x in fallbacks["transcription"]
            ]
        if "vision" in fallbacks and isinstance(fallbacks["vision"], list):
            config.vision_fallbacks = [str(x) for x in fallbacks["vision"]]
        if "reasoning" in fallbacks and isinstance(fallbacks["reasoning"], list):
            config.reasoning_fallbacks = [str(x) for x in fallbacks["reasoning"]]

    # Load parallel_fallback settings
    parallel = providers_section.get("parallel_fallback", {})
    if isinstance(parallel, dict):
        for key, value in parallel.items():
            if isinstance(value, bool):
                config.parallel_fallback[str(key)] = value

    # Apply legacy env var fallbacks
    _apply_legacy_env_vars(config)

    return config


# Singleton for global config
_config: ProvidersConfig | None = None


def get_providers_config() -> ProvidersConfig:
    """Get the global providers configuration.

    Loads from config file on first call, then returns cached.
    Uses the same config file discovery as the base config loader.

    Returns:
        ProvidersConfig with all settings resolved.
    """
    global _config
    if _config is not None:
        return _config

    from claudetube.config.loader import (
        _find_project_config,
        _get_user_config_path,
        _load_yaml_config,
    )

    yaml_config: dict | None = None

    # Try project config first, then user config
    project_path = _find_project_config()
    if project_path:
        yaml_config = _load_yaml_config(project_path)

    if yaml_config is None:
        user_path = _get_user_config_path()
        yaml_config = _load_yaml_config(user_path)

    _config = load_providers_config(yaml_config)
    return _config


def clear_providers_config_cache() -> None:
    """Clear cached providers config (for testing or config reload).

    After clearing, the next get_providers_config() call will reload
    from config files and environment.
    """
    global _config
    _config = None


__all__ = [
    "ProviderConfig",
    "ProvidersConfig",
    "ConfigValidationResult",
    "validate_providers_config",
    "load_providers_config",
    "get_providers_config",
    "clear_providers_config_cache",
]
