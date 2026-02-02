"""Integration tests for provider routing end-to-end flows.

Tests the full routing pipeline: config -> router -> provider selection
-> capability validation. Uses mock providers to avoid real API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from claudetube.providers.capabilities import Capability, ProviderInfo
from claudetube.providers.config import ProviderConfig, ProvidersConfig
from claudetube.providers.router import NoProviderError, ProviderRouter


def _make_mock_provider(name, capabilities, available=True):
    """Create a mock provider with specified capabilities."""
    mock = MagicMock()
    mock.info = ProviderInfo(
        name=name,
        capabilities=frozenset(capabilities),
    )
    mock.is_available.return_value = available

    # Add protocol methods based on capabilities
    if Capability.TRANSCRIBE in capabilities:
        mock.transcribe = MagicMock()
    if Capability.VISION in capabilities:
        mock.analyze_images = MagicMock()
    if Capability.VIDEO in capabilities:
        mock.analyze_video = MagicMock()
    if Capability.REASON in capabilities:
        mock.reason = MagicMock()
    if Capability.EMBED in capabilities:
        mock.embed = MagicMock()

    return mock


class TestRoutingWithDefaultConfig:
    """Test routing behavior with default configuration."""

    def test_default_transcription_uses_whisper_local(self):
        """Default config routes transcription to whisper-local."""
        config = ProvidersConfig()
        assert config.transcription_provider == "whisper-local"

        mock_whisper = _make_mock_provider("whisper-local", {Capability.TRANSCRIBE})

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_whisper,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.TRANSCRIBE)

        assert provider is mock_whisper

    def test_default_vision_uses_claude_code(self):
        """Default config routes vision to claude-code."""
        config = ProvidersConfig()
        assert config.vision_provider == "claude-code"

        mock_claude = _make_mock_provider(
            "claude-code", {Capability.VISION, Capability.REASON}
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_claude,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.VISION)

        assert provider is mock_claude

    def test_default_reasoning_uses_claude_code(self):
        """Default config routes reasoning to claude-code."""
        config = ProvidersConfig()
        assert config.reasoning_provider == "claude-code"

        mock_claude = _make_mock_provider(
            "claude-code", {Capability.VISION, Capability.REASON}
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_claude,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.REASON)

        assert provider is mock_claude


class TestRoutingWithCustomConfig:
    """Test routing with user-specified preferences."""

    def test_custom_transcription_preference(self):
        """Config can override transcription to OpenAI."""
        config = ProvidersConfig(
            transcription_provider="openai",
            providers={
                "openai": ProviderConfig(api_key="sk-test"),
            },
        )

        mock_openai = _make_mock_provider(
            "openai",
            {Capability.TRANSCRIBE, Capability.VISION, Capability.REASON},
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_openai,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.TRANSCRIBE)

        assert provider is mock_openai

    def test_custom_vision_preference(self):
        """Config can override vision to Anthropic."""
        config = ProvidersConfig(
            vision_provider="anthropic",
            providers={
                "anthropic": ProviderConfig(api_key="sk-ant"),
            },
        )

        mock_anthropic = _make_mock_provider(
            "anthropic", {Capability.VISION, Capability.REASON}
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_anthropic,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.VISION)

        assert provider is mock_anthropic


class TestFallbackChains:
    """Test fallback behavior when preferred provider is unavailable."""

    def test_fallback_on_unavailable_preferred(self):
        """Should fall back when preferred provider is unavailable."""
        config = ProvidersConfig(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
        )

        mock_unavailable = _make_mock_provider(
            "anthropic", {Capability.VISION}, available=False
        )
        mock_openai = _make_mock_provider(
            "openai", {Capability.VISION, Capability.REASON}
        )

        def _mock_get_provider(name, **kwargs):
            if name == "anthropic":
                return mock_unavailable
            if name == "openai":
                return mock_openai
            return mock_unavailable

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.VISION)

        assert provider is mock_openai

    def test_fallback_on_import_error(self):
        """Should fall back when preferred provider fails to import."""
        config = ProvidersConfig(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
        )

        mock_openai = _make_mock_provider(
            "openai", {Capability.VISION, Capability.REASON}
        )

        def _mock_get_provider(name, **kwargs):
            if name == "anthropic":
                raise ImportError("anthropic not installed")
            if name == "openai":
                return mock_openai
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.VISION)

        assert provider is mock_openai

    def test_claude_code_ultimate_fallback_for_vision(self):
        """Claude-code should be ultimate fallback for vision."""
        config = ProvidersConfig(
            vision_provider="anthropic",
            vision_fallbacks=["openai"],
        )

        mock_claude = _make_mock_provider(
            "claude-code", {Capability.VISION, Capability.REASON}
        )

        def _mock_get_provider(name, **kwargs):
            if name == "claude-code":
                return mock_claude
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.VISION)

        assert provider is mock_claude

    def test_claude_code_ultimate_fallback_for_reasoning(self):
        """Claude-code should be ultimate fallback for reasoning."""
        config = ProvidersConfig(
            reasoning_provider="openai",
            reasoning_fallbacks=[],
        )

        mock_claude = _make_mock_provider(
            "claude-code", {Capability.VISION, Capability.REASON}
        )

        def _mock_get_provider(name, **kwargs):
            if name == "claude-code":
                return mock_claude
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            provider = router.get_for_capability(Capability.REASON)

        assert provider is mock_claude

    def test_no_claude_code_fallback_for_transcription(self):
        """Claude-code should NOT be fallback for transcription."""
        config = ProvidersConfig(
            transcription_provider="openai",
            transcription_fallbacks=[],
        )

        def _mock_get_provider(name, **kwargs):
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            with pytest.raises(NoProviderError):
                router.get_for_capability(Capability.TRANSCRIBE)

    def test_no_provider_error_when_all_fail(self):
        """Should raise NoProviderError when all providers fail."""
        config = ProvidersConfig(
            embedding_provider="voyage",
        )

        def _mock_get_provider(name, **kwargs):
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            with pytest.raises(NoProviderError) as exc_info:
                router.get_for_capability(Capability.EMBED)

            assert exc_info.value.capability == Capability.EMBED


class TestCallWithFallback:
    """Test call_with_fallback for automatic retries and failover."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Should succeed on first try with available provider."""
        config = ProvidersConfig(vision_provider="anthropic")

        mock_provider = _make_mock_provider(
            "anthropic", {Capability.VISION, Capability.REASON}
        )
        mock_provider.analyze_images = MagicMock(return_value="description")

        # Make analyze_images a coroutine

        async def mock_analyze(*args, **kwargs):
            return "scene description"

        mock_provider.analyze_images = mock_analyze

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            router = ProviderRouter(config=config)
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["frame.jpg"],
                prompt="Describe this",
            )

        assert result == "scene description"

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """Should fall back to next provider on error."""
        config = ProvidersConfig(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
        )

        mock_anthropic = _make_mock_provider("anthropic", {Capability.VISION})

        async def _fail(*args, **kwargs):
            raise RuntimeError("API error")

        mock_anthropic.analyze_images = _fail

        mock_openai = _make_mock_provider(
            "openai", {Capability.VISION, Capability.REASON}
        )

        async def _succeed(*args, **kwargs):
            return "openai result"

        mock_openai.analyze_images = _succeed

        def _mock_get_provider(name, **kwargs):
            if name == "anthropic":
                return mock_anthropic
            if name == "openai":
                return mock_openai
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            router = ProviderRouter(config=config)
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["frame.jpg"],
                prompt="Describe this",
            )

        assert result == "openai result"


class TestConfigToRouterIntegration:
    """Test loading config and using it with router."""

    def test_load_config_and_route(self, monkeypatch):
        """Full flow: YAML config -> ProvidersConfig -> ProviderRouter."""
        from claudetube.providers.config import load_providers_config

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

        monkeypatch.setenv("MY_KEY", "sk-test")

        yaml_config = {
            "providers": {
                "openai": {"api_key": "${MY_KEY}", "model": "gpt-4o"},
                "preferences": {
                    "transcription": "openai",
                    "vision": "claude-code",
                },
                "fallbacks": {
                    "vision": ["openai", "claude-code"],
                },
            }
        }

        config = load_providers_config(yaml_config)

        assert config.transcription_provider == "openai"
        assert config.vision_provider == "claude-code"
        assert config.vision_fallbacks == ["openai", "claude-code"]
        assert config.providers["openai"].api_key == "sk-test"

        # Verify router can use this config
        router = ProviderRouter(config=config)
        assert router._config is config

    def test_default_config_has_safe_defaults(self, monkeypatch):
        """Default config should work without any API keys."""
        from claudetube.providers.config import load_providers_config

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

        # Defaults should point to always-available providers
        assert config.transcription_provider == "whisper-local"
        assert config.vision_provider == "claude-code"
        assert config.reasoning_provider == "claude-code"
