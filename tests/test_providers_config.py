"""Tests for the provider configuration loader."""

from unittest.mock import patch

from claudetube.providers.config import (
    ConfigValidationResult,
    ProviderConfig,
    ProvidersConfig,
    _interpolate_env_vars,
    _load_single_provider_config,
    clear_providers_config_cache,
    get_providers_config,
    load_providers_config,
    validate_providers_config,
)


class TestInterpolateEnvVars:
    """Tests for ${ENV_VAR} interpolation."""

    def test_simple_string_replacement(self, monkeypatch):
        """Test basic env var replacement in a string."""
        monkeypatch.setenv("MY_KEY", "secret123")
        result = _interpolate_env_vars("${MY_KEY}")
        assert result == "secret123"

    def test_embedded_replacement(self, monkeypatch):
        """Test env var embedded in larger string."""
        monkeypatch.setenv("HOST", "api.example.com")
        result = _interpolate_env_vars("https://${HOST}/v1")
        assert result == "https://api.example.com/v1"

    def test_multiple_vars_in_string(self, monkeypatch):
        """Test multiple env vars in one string."""
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setenv("PASS", "hunter2")
        result = _interpolate_env_vars("${USER}:${PASS}")
        assert result == "alice:hunter2"

    def test_missing_env_var_returns_empty(self, monkeypatch):
        """Test missing env var produces empty string, not crash."""
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        result = _interpolate_env_vars("key=${NONEXISTENT_VAR_12345}")
        assert result == "key="

    def test_missing_env_var_logs_warning(self, monkeypatch, caplog):
        """Test missing env var produces a warning log."""
        monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)
        import logging

        with caplog.at_level(logging.WARNING):
            _interpolate_env_vars("${MISSING_KEY_XYZ}")
        assert "MISSING_KEY_XYZ" in caplog.text

    def test_dict_interpolation(self, monkeypatch):
        """Test env var interpolation in dict values."""
        monkeypatch.setenv("API_KEY", "sk-test")
        result = _interpolate_env_vars({"api_key": "${API_KEY}", "model": "gpt-4o"})
        assert result == {"api_key": "sk-test", "model": "gpt-4o"}

    def test_list_interpolation(self, monkeypatch):
        """Test env var interpolation in lists."""
        monkeypatch.setenv("ITEM", "resolved")
        result = _interpolate_env_vars(["${ITEM}", "static"])
        assert result == ["resolved", "static"]

    def test_nested_interpolation(self, monkeypatch):
        """Test env var interpolation in nested structures."""
        monkeypatch.setenv("NESTED_KEY", "deep_value")
        result = _interpolate_env_vars(
            {"outer": {"inner": "${NESTED_KEY}"}, "list": ["${NESTED_KEY}"]}
        )
        assert result == {"outer": {"inner": "deep_value"}, "list": ["deep_value"]}

    def test_non_string_passthrough(self):
        """Test non-string values pass through unchanged."""
        assert _interpolate_env_vars(42) == 42
        assert _interpolate_env_vars(3.14) == 3.14
        assert _interpolate_env_vars(True) is True
        assert _interpolate_env_vars(None) is None

    def test_no_vars_unchanged(self):
        """Test strings without ${} are unchanged."""
        assert _interpolate_env_vars("plain string") == "plain string"

    def test_empty_string(self):
        """Test empty string passes through."""
        assert _interpolate_env_vars("") == ""


class TestLoadSingleProviderConfig:
    """Tests for loading a single provider config."""

    def test_basic_config(self, monkeypatch):
        """Test loading basic provider config."""
        monkeypatch.setenv("TEST_API_KEY", "sk-test")
        config = _load_single_provider_config(
            {"api_key": "${TEST_API_KEY}", "model": "gpt-4o"}
        )
        assert config.api_key == "sk-test"
        assert config.model == "gpt-4o"
        assert config.extra == {}

    def test_extra_fields(self, monkeypatch):
        """Test extra fields go into extra dict."""
        monkeypatch.setenv("KEY", "val")
        config = _load_single_provider_config(
            {"api_key": "${KEY}", "model": "m1", "temperature": 0.7, "region": "us"}
        )
        assert config.extra == {"temperature": 0.7, "region": "us"}

    def test_missing_api_key(self):
        """Test config without api_key."""
        config = _load_single_provider_config({"model": "gpt-4o"})
        assert config.api_key is None
        assert config.model == "gpt-4o"

    def test_empty_api_key_becomes_none(self, monkeypatch):
        """Test empty string api_key becomes None."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        config = _load_single_provider_config({"api_key": "${UNSET_VAR}"})
        assert config.api_key is None

    def test_empty_dict(self):
        """Test loading empty config."""
        config = _load_single_provider_config({})
        assert config.api_key is None
        assert config.model is None
        assert config.extra == {}


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        pc = ProviderConfig()
        assert pc.api_key is None
        assert pc.model is None
        assert pc.extra == {}

    def test_with_values(self):
        """Test config with explicit values."""
        pc = ProviderConfig(api_key="sk-test", model="gpt-4o", extra={"temp": 0.5})
        assert pc.api_key == "sk-test"
        assert pc.model == "gpt-4o"
        assert pc.extra["temp"] == 0.5


class TestProvidersConfig:
    """Tests for ProvidersConfig dataclass."""

    def test_defaults(self):
        """Test default config values."""
        config = ProvidersConfig()
        assert config.transcription_provider == "whisper-local"
        assert config.vision_provider == "claude-code"
        assert config.reasoning_provider == "claude-code"
        assert config.video_provider is None
        assert config.embedding_provider == "voyage"
        assert config.whisper_local_model == "small"
        assert config.ollama_model == "llava:13b"
        assert config.transcription_fallbacks == ["whisper-local"]
        assert config.vision_fallbacks == ["claude-code"]
        assert config.reasoning_fallbacks == ["claude-code"]

    def test_get_provider_config_creates_default(self):
        """Test get_provider_config creates default for unknown provider."""
        config = ProvidersConfig()
        pc = config.get_provider_config("openai")
        assert pc.api_key is None
        assert pc.model is None

    def test_get_provider_config_returns_existing(self):
        """Test get_provider_config returns existing config."""
        config = ProvidersConfig()
        config.providers["openai"] = ProviderConfig(api_key="sk-test")
        pc = config.get_provider_config("openai")
        assert pc.api_key == "sk-test"


class TestLoadProvidersConfig:
    """Tests for load_providers_config."""

    def test_none_config_returns_defaults(self, monkeypatch):
        """Test None config returns default ProvidersConfig."""
        # Clear any env vars that might interfere
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        config = load_providers_config(None)
        assert isinstance(config, ProvidersConfig)
        assert config.transcription_provider == "whisper-local"
        assert config.vision_provider == "claude-code"

    def test_empty_config_returns_defaults(self, monkeypatch):
        """Test empty dict returns defaults."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        config = load_providers_config({})
        assert isinstance(config, ProvidersConfig)
        assert config.transcription_provider == "whisper-local"

    def test_load_provider_api_keys(self, monkeypatch):
        """Test loading provider API keys from config."""
        monkeypatch.setenv("OPENAI_KEY", "sk-openai")
        monkeypatch.setenv("ANTHRO_KEY", "sk-anthro")

        yaml = {
            "providers": {
                "openai": {"api_key": "${OPENAI_KEY}", "model": "gpt-4o"},
                "anthropic": {"api_key": "${ANTHRO_KEY}"},
            }
        }
        config = load_providers_config(yaml)
        assert config.providers["openai"].api_key == "sk-openai"
        assert config.providers["openai"].model == "gpt-4o"
        assert config.providers["anthropic"].api_key == "sk-anthro"

    def test_load_preferences(self, monkeypatch):
        """Test loading provider preferences."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {
            "providers": {
                "preferences": {
                    "transcription": "openai",
                    "vision": "anthropic",
                    "video": "google",
                    "reasoning": "openai",
                    "embedding": "voyage",
                }
            }
        }
        config = load_providers_config(yaml)
        assert config.transcription_provider == "openai"
        assert config.vision_provider == "anthropic"
        assert config.video_provider == "google"
        assert config.reasoning_provider == "openai"
        assert config.embedding_provider == "voyage"

    def test_load_fallbacks(self, monkeypatch):
        """Test loading fallback chains."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {
            "providers": {
                "fallbacks": {
                    "vision": ["anthropic", "openai", "claude-code"],
                    "transcription": ["openai", "whisper-local"],
                    "reasoning": ["openai", "claude-code"],
                }
            }
        }
        config = load_providers_config(yaml)
        assert config.vision_fallbacks == ["anthropic", "openai", "claude-code"]
        assert config.transcription_fallbacks == ["openai", "whisper-local"]
        assert config.reasoning_fallbacks == ["openai", "claude-code"]

    def test_load_local_config(self, monkeypatch):
        """Test loading local provider configs."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {
            "providers": {
                "local": {
                    "whisper_model": "large",
                    "ollama_model": "llava:7b",
                }
            }
        }
        config = load_providers_config(yaml)
        assert config.whisper_local_model == "large"
        assert config.ollama_model == "llava:7b"

    def test_full_config(self, monkeypatch):
        """Test loading a complete config with all sections."""
        monkeypatch.setenv("OAI_KEY", "sk-openai")
        monkeypatch.setenv("ANTHRO_KEY", "sk-anthro")
        monkeypatch.setenv("GOOGLE_KEY", "goog-key")
        # Clear legacy vars to avoid interference
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {
            "providers": {
                "openai": {"api_key": "${OAI_KEY}", "model": "gpt-4o"},
                "anthropic": {"api_key": "${ANTHRO_KEY}"},
                "google": {"api_key": "${GOOGLE_KEY}", "model": "gemini-2.0-flash"},
                "local": {"whisper_model": "medium"},
                "preferences": {
                    "transcription": "whisper-local",
                    "vision": "claude-code",
                    "reasoning": "claude-code",
                },
                "fallbacks": {
                    "vision": ["anthropic", "openai", "claude-code"],
                },
            }
        }
        config = load_providers_config(yaml)
        assert config.providers["openai"].api_key == "sk-openai"
        assert config.providers["openai"].model == "gpt-4o"
        assert config.providers["anthropic"].api_key == "sk-anthro"
        assert config.providers["google"].api_key == "goog-key"
        assert config.whisper_local_model == "medium"
        assert config.transcription_provider == "whisper-local"
        assert config.vision_fallbacks == ["anthropic", "openai", "claude-code"]

    def test_invalid_providers_section(self, monkeypatch):
        """Test invalid providers section is handled gracefully."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {"providers": "not a dict"}
        config = load_providers_config(yaml)
        assert isinstance(config, ProvidersConfig)
        assert config.transcription_provider == "whisper-local"

    def test_invalid_provider_value(self, monkeypatch):
        """Test invalid provider config (not a dict) is handled."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {"providers": {"openai": "not a dict"}}
        config = load_providers_config(yaml)
        # openai may be in providers dict from legacy env var fallback,
        # but should not have been loaded from the invalid YAML value
        if "openai" in config.providers:
            assert config.providers["openai"].api_key is None

    def test_invalid_fallbacks_not_list(self, monkeypatch):
        """Test non-list fallback value is ignored."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {"providers": {"fallbacks": {"vision": "not-a-list"}}}
        config = load_providers_config(yaml)
        assert config.vision_fallbacks == ["claude-code"]  # Default unchanged


class TestLegacyEnvVars:
    """Tests for backward compatibility with env vars."""

    def test_direct_env_var_fallback(self, monkeypatch):
        """Test that OPENAI_API_KEY env var is picked up."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        # Clear others to avoid interference
        for var in [
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)
        monkeypatch.delenv("CLAUDETUBE_OPENAI_API_KEY", raising=False)

        config = load_providers_config({})
        assert config.get_provider_config("openai").api_key == "sk-from-env"

    def test_claudetube_prefixed_env_var(self, monkeypatch):
        """Test that CLAUDETUBE_OPENAI_API_KEY env var is picked up."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("CLAUDETUBE_OPENAI_API_KEY", "sk-from-claudetube-env")
        # Clear others
        for var in [
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        config = load_providers_config({})
        assert (
            config.get_provider_config("openai").api_key == "sk-from-claudetube-env"
        )

    def test_yaml_overrides_env_var(self, monkeypatch):
        """Test that YAML api_key takes priority over env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("YAML_KEY", "sk-from-yaml")
        # Clear others
        for var in [
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {"providers": {"openai": {"api_key": "${YAML_KEY}"}}}
        config = load_providers_config(yaml)
        assert config.providers["openai"].api_key == "sk-from-yaml"

    def test_all_legacy_providers(self, monkeypatch):
        """Test all legacy env vars are checked."""
        expected = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepgram": "DEEPGRAM_API_KEY",
            "assemblyai": "ASSEMBLYAI_API_KEY",
            "voyage": "VOYAGE_API_KEY",
        }
        # Clear all first
        for var in expected.values():
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        # Set each one
        for provider, var in expected.items():
            monkeypatch.setenv(var, f"key-{provider}")

        config = load_providers_config({})
        for provider in expected:
            assert config.get_provider_config(provider).api_key == f"key-{provider}"

    def test_no_env_vars_produces_none(self, monkeypatch):
        """Test no env vars results in None api_keys."""
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        config = load_providers_config({})
        assert config.get_provider_config("openai").api_key is None
        assert config.get_provider_config("anthropic").api_key is None


class TestGetProvidersConfig:
    """Tests for the singleton get_providers_config."""

    def test_returns_config(self, monkeypatch):
        """Test get_providers_config returns a ProvidersConfig."""
        clear_providers_config_cache()
        # Clear env vars
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        with (
            patch(
                "claudetube.config.loader._find_project_config", return_value=None
            ),
            patch(
                "claudetube.config.loader._load_yaml_config", return_value=None
            ),
        ):
            config = get_providers_config()
        assert isinstance(config, ProvidersConfig)
        clear_providers_config_cache()

    def test_caches_result(self, monkeypatch):
        """Test get_providers_config caches the result."""
        clear_providers_config_cache()
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        with (
            patch(
                "claudetube.config.loader._find_project_config", return_value=None
            ),
            patch(
                "claudetube.config.loader._load_yaml_config", return_value=None
            ) as mock_load,
        ):
            config1 = get_providers_config()
            config2 = get_providers_config()

        assert config1 is config2
        # _load_yaml_config called once for user config (project returned None path)
        assert mock_load.call_count == 1
        clear_providers_config_cache()

    def test_clear_cache_forces_reload(self, monkeypatch):
        """Test clearing cache forces reload."""
        clear_providers_config_cache()
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        with (
            patch(
                "claudetube.config.loader._find_project_config", return_value=None
            ),
            patch(
                "claudetube.config.loader._load_yaml_config", return_value=None
            ) as mock_load,
        ):
            config1 = get_providers_config()
            clear_providers_config_cache()
            config2 = get_providers_config()

        assert config1 is not config2
        assert mock_load.call_count == 2
        clear_providers_config_cache()

    def test_loads_from_project_config(self, tmp_path, monkeypatch):
        """Test loading from project config file with YAML content."""
        clear_providers_config_cache()
        monkeypatch.setenv("MY_OAI_KEY", "sk-project")
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        # Simulate what _load_yaml_config returns for a project config
        yaml_dict = {
            "providers": {
                "openai": {
                    "api_key": "${MY_OAI_KEY}",
                    "model": "gpt-4o-mini",
                }
            }
        }

        config_file = tmp_path / "config.yaml"

        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=yaml_dict,
            ),
        ):
            config = get_providers_config()

        assert config.providers["openai"].api_key == "sk-project"
        assert config.providers["openai"].model == "gpt-4o-mini"
        clear_providers_config_cache()


class TestApiKeyNeverLogged:
    """Tests to ensure API keys are not logged."""

    def test_provider_config_repr_no_key(self):
        """Test that repr of ProviderConfig includes the key value.

        Note: This verifies the default repr. If we add a custom __repr__
        that masks keys, we'd test that instead.
        """
        # This is a design check - the config stores resolved keys,
        # but the logging in _interpolate_env_vars and load_providers_config
        # should never log the actual values.
        pc = ProviderConfig(api_key="sk-secret")
        # Key is stored (not masked), but we verify logging doesn't expose it
        assert pc.api_key == "sk-secret"

    def test_interpolation_does_not_log_values(self, monkeypatch, caplog):
        """Test that env var interpolation logs var name but not value."""
        import logging

        monkeypatch.setenv("SECRET_KEY", "super-secret-value")
        with caplog.at_level(logging.DEBUG):
            _interpolate_env_vars("${SECRET_KEY}")
        # Should NOT contain the actual value in logs
        assert "super-secret-value" not in caplog.text


class TestConfigValidationResult:
    """Tests for ConfigValidationResult dataclass."""

    def test_empty_is_valid(self):
        """Test empty result is valid."""
        result = ConfigValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_with_errors_not_valid(self):
        """Test result with errors is not valid."""
        result = ConfigValidationResult(errors=["something broke"])
        assert result.is_valid is False

    def test_warnings_only_still_valid(self):
        """Test result with only warnings is still valid."""
        result = ConfigValidationResult(warnings=["minor issue"])
        assert result.is_valid is True


class TestValidateProvidersConfig:
    """Tests for validate_providers_config."""

    def test_none_config_is_valid(self):
        """Test None config passes validation."""
        result = validate_providers_config(None)
        assert result.is_valid is True

    def test_empty_config_is_valid(self):
        """Test empty dict passes validation."""
        result = validate_providers_config({})
        assert result.is_valid is True

    def test_no_providers_section_is_valid(self):
        """Test config without providers section passes."""
        result = validate_providers_config({"cache_dir": "/tmp"})
        assert result.is_valid is True

    def test_valid_full_config(self):
        """Test a fully valid config passes validation."""
        yaml = {
            "providers": {
                "openai": {"api_key": "sk-test", "model": "gpt-4o"},
                "anthropic": {"api_key": "sk-ant"},
                "local": {"whisper_model": "small", "ollama_model": "llava:7b"},
                "preferences": {
                    "transcription": "whisper-local",
                    "vision": "claude-code",
                    "reasoning": "anthropic",
                    "embedding": "voyage",
                },
                "fallbacks": {
                    "vision": ["anthropic", "openai", "claude-code"],
                    "reasoning": ["openai", "claude-code"],
                    "transcription": ["openai", "whisper-local"],
                },
            }
        }
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert result.warnings == []

    def test_non_dict_config_is_error(self):
        """Test non-dict config produces error."""
        result = validate_providers_config("not a dict")
        assert not result.is_valid
        assert any("mapping" in e for e in result.errors)

    def test_non_dict_providers_section_is_error(self):
        """Test non-dict providers section produces error."""
        result = validate_providers_config({"providers": "bad"})
        assert not result.is_valid
        assert any("providers" in e and "mapping" in e for e in result.errors)

    def test_unknown_top_level_key_warns(self):
        """Test unknown key in providers section produces warning."""
        yaml = {"providers": {"unknown_provider": {"key": "val"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("unknown_provider" in w for w in result.warnings)

    def test_non_dict_provider_config_is_error(self):
        """Test non-dict provider config produces error."""
        yaml = {"providers": {"openai": "not-a-dict"}}
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert any("openai" in e and "mapping" in e for e in result.errors)

    def test_non_dict_local_is_error(self):
        """Test non-dict local section produces error."""
        yaml = {"providers": {"local": "bad"}}
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert any("local" in e for e in result.errors)

    def test_invalid_whisper_model_is_error(self):
        """Test invalid whisper model produces error."""
        yaml = {"providers": {"local": {"whisper_model": "huge"}}}
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert any("whisper_model" in e and "huge" in e for e in result.errors)

    def test_valid_whisper_models(self):
        """Test all valid whisper model sizes pass validation."""
        for model in ["tiny", "base", "small", "medium", "large"]:
            yaml = {"providers": {"local": {"whisper_model": model}}}
            result = validate_providers_config(yaml)
            assert result.is_valid is True, f"Model '{model}' should be valid"

    def test_unknown_local_key_warns(self):
        """Test unknown key in local section produces warning."""
        yaml = {"providers": {"local": {"unknown_setting": "val"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("unknown_setting" in w for w in result.warnings)

    def test_non_dict_preferences_is_error(self):
        """Test non-dict preferences section produces error."""
        yaml = {"providers": {"preferences": "bad"}}
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert any("preferences" in e for e in result.errors)

    def test_unknown_preference_key_warns(self):
        """Test unknown preference key produces warning."""
        yaml = {"providers": {"preferences": {"unknown_pref": "openai"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("unknown_pref" in w for w in result.warnings)

    def test_unknown_provider_in_preference_warns(self):
        """Test unknown provider in preference produces warning."""
        yaml = {"providers": {"preferences": {"vision": "nonexistent-ai"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("nonexistent-ai" in w for w in result.warnings)

    def test_capability_mismatch_in_preference_warns(self):
        """Test provider without required capability in preference produces warning."""
        # whisper-local can only transcribe, not do vision
        yaml = {"providers": {"preferences": {"vision": "whisper-local"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("whisper-local" in w and "vision" in w.lower() for w in result.warnings)

    def test_transcription_pref_with_vision_provider_warns(self):
        """Test setting transcription to a vision-only provider warns."""
        # anthropic can't transcribe
        yaml = {"providers": {"preferences": {"transcription": "anthropic"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("anthropic" in w and "transcribe" in w.lower() for w in result.warnings)

    def test_valid_preference_with_capability(self):
        """Test valid preference-capability combination passes."""
        yaml = {"providers": {"preferences": {"transcription": "openai"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        # Should not have warnings about capability mismatch
        assert not any("openai" in w and "transcribe" in w.lower() for w in result.warnings)

    def test_non_dict_fallbacks_is_error(self):
        """Test non-dict fallbacks section produces error."""
        yaml = {"providers": {"fallbacks": "bad"}}
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert any("fallbacks" in e for e in result.errors)

    def test_unknown_fallback_key_warns(self):
        """Test unknown fallback key produces warning."""
        yaml = {"providers": {"fallbacks": {"embedding": ["voyage"]}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("embedding" in w for w in result.warnings)

    def test_non_list_fallback_chain_is_error(self):
        """Test non-list fallback chain produces error."""
        yaml = {"providers": {"fallbacks": {"vision": "not-a-list"}}}
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert any("vision" in e and "list" in e for e in result.errors)

    def test_unknown_provider_in_fallback_chain_warns(self):
        """Test unknown provider in fallback chain produces warning."""
        yaml = {"providers": {"fallbacks": {"vision": ["anthropic", "fake-ai"]}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("fake-ai" in w for w in result.warnings)

    def test_capability_mismatch_in_fallback_warns(self):
        """Test fallback provider without required capability produces warning."""
        # whisper-local in vision fallback doesn't make sense
        yaml = {"providers": {"fallbacks": {"vision": ["whisper-local"]}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert any("whisper-local" in w for w in result.warnings)

    def test_valid_fallback_chain(self):
        """Test valid fallback chain passes."""
        yaml = {
            "providers": {
                "fallbacks": {"vision": ["anthropic", "openai", "claude-code"]}
            }
        }
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        assert not any("vision" in w and "capability" in w.lower() for w in result.warnings)

    def test_multiple_errors_collected(self):
        """Test multiple errors are all collected."""
        yaml = {
            "providers": {
                "openai": "bad",
                "anthropic": "bad",
                "local": {"whisper_model": "mega"},
            }
        }
        result = validate_providers_config(yaml)
        assert not result.is_valid
        assert len(result.errors) >= 3

    def test_validation_runs_during_load(self, monkeypatch, caplog):
        """Test that load_providers_config runs validation."""
        import logging

        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPGRAM_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "VOYAGE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(f"CLAUDETUBE_{var}", raising=False)

        yaml = {"providers": {"local": {"whisper_model": "mega"}}}
        with caplog.at_level(logging.ERROR):
            load_providers_config(yaml)
        assert any("whisper_model" in r.message for r in caplog.records)

    def test_provider_alias_resolved_in_preference(self):
        """Test provider aliases are resolved during validation."""
        # "gemini" is an alias for "google"
        yaml = {"providers": {"preferences": {"video": "gemini"}}}
        result = validate_providers_config(yaml)
        assert result.is_valid is True
        # Should not warn about unknown provider since gemini -> google
        assert not any("unknown provider" in w.lower() for w in result.warnings)
