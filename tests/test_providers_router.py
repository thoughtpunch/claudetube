"""Tests for the provider router with fallback chains.

Verifies that:
1. Preferred provider selection works for each capability
2. Fallback chain activates when preferred is unavailable
3. Claude-code serves as ultimate fallback for VISION/REASON
4. NoProviderError raised when nothing works for TRANSCRIBE
5. call_with_fallback retries on rate limits, then falls back
6. Convenience methods (get_transcriber, etc.) work correctly
7. Custom and default config are handled
8. Logging of provider selection is correct
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.capabilities import Capability, CostTier, ProviderInfo
from claudetube.providers.config import ProvidersConfig
from claudetube.providers.router import (
    NoProviderError,
    ProviderRouter,
    _is_rate_limit_error,
)

# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_mock_provider(
    name: str,
    capabilities: frozenset[Capability],
    available: bool = True,
) -> MagicMock:
    """Create a mock provider with given capabilities.

    The mock implements Provider, and optionally Transcriber, VisionAnalyzer,
    VideoAnalyzer, Reasoner, and Embedder protocols based on capabilities.
    """
    from claudetube.providers.base import (
        Embedder,
        Reasoner,
        Transcriber,
        VideoAnalyzer,
        VisionAnalyzer,
    )

    # Determine which protocols to spec from
    specs = []
    if Capability.TRANSCRIBE in capabilities:
        specs.append(Transcriber)
    if Capability.VISION in capabilities:
        specs.append(VisionAnalyzer)
    if Capability.VIDEO in capabilities:
        specs.append(VideoAnalyzer)
    if Capability.REASON in capabilities:
        specs.append(Reasoner)
    if Capability.EMBED in capabilities:
        specs.append(Embedder)

    mock = MagicMock()
    mock.info = ProviderInfo(name=name, capabilities=capabilities)
    mock.is_available.return_value = available

    # Make isinstance checks work for protocol checking
    # We use __class__ patching to make isinstance work
    if Capability.TRANSCRIBE in capabilities:
        mock.transcribe = AsyncMock(return_value="transcription result")
    if Capability.VISION in capabilities:
        mock.analyze_images = AsyncMock(return_value="vision result")
    if Capability.VIDEO in capabilities:
        mock.analyze_video = AsyncMock(return_value="video result")
    if Capability.REASON in capabilities:
        mock.reason = AsyncMock(return_value="reasoning result")
    if Capability.EMBED in capabilities:
        mock.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    return mock


def _make_config(**overrides) -> ProvidersConfig:
    """Create a ProvidersConfig with optional overrides."""
    defaults = {
        "transcription_provider": "whisper-local",
        "vision_provider": "claude-code",
        "video_provider": None,
        "reasoning_provider": "claude-code",
        "embedding_provider": "voyage",
        "cost_preference": "cost",
        "transcription_fallbacks": ["whisper-local"],
        "vision_fallbacks": ["claude-code"],
        "reasoning_fallbacks": ["claude-code"],
    }
    defaults.update(overrides)
    return ProvidersConfig(**defaults)


# =============================================================================
# NoProviderError tests
# =============================================================================


class TestNoProviderError:
    """Tests for the NoProviderError exception."""

    def test_stores_capability(self):
        """NoProviderError stores the capability that failed."""
        err = NoProviderError(Capability.TRANSCRIBE)
        assert err.capability == Capability.TRANSCRIBE

    def test_default_message(self):
        """NoProviderError has a useful default message."""
        err = NoProviderError(Capability.VISION)
        assert "VISION" in str(err)
        assert "No available provider" in str(err)

    def test_custom_message(self):
        """NoProviderError accepts a custom message."""
        err = NoProviderError(Capability.EMBED, "Custom error message")
        assert str(err) == "Custom error message"
        assert err.capability == Capability.EMBED

    def test_is_exception(self):
        """NoProviderError is a proper Exception subclass."""
        err = NoProviderError(Capability.REASON)
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        """NoProviderError can be raised and caught."""
        with pytest.raises(NoProviderError) as exc_info:
            raise NoProviderError(Capability.TRANSCRIBE)
        assert exc_info.value.capability == Capability.TRANSCRIBE


# =============================================================================
# ProviderRouter.__init__ tests
# =============================================================================


class TestProviderRouterInit:
    """Tests for ProviderRouter initialization."""

    def test_with_custom_config(self):
        """ProviderRouter accepts a custom ProvidersConfig."""
        config = _make_config(vision_provider="anthropic")
        router = ProviderRouter(config=config)
        assert router._config.vision_provider == "anthropic"

    def test_with_default_config(self):
        """ProviderRouter loads default config when None is passed."""
        with patch(
            "claudetube.providers.config.get_providers_config"
        ) as mock_get:
            mock_get.return_value = _make_config()
            router = ProviderRouter(config=None)
            mock_get.assert_called_once()
            assert router._config.transcription_provider == "whisper-local"


# =============================================================================
# get_for_capability tests
# =============================================================================


class TestGetForCapability:
    """Tests for ProviderRouter.get_for_capability."""

    def test_returns_preferred_transcription_provider(self):
        """Preferred transcription provider is returned when available."""
        config = _make_config(transcription_provider="whisper-local")
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_for_capability(Capability.TRANSCRIBE)
        assert result is mock_provider

    def test_returns_preferred_vision_provider(self):
        """Preferred vision provider is returned when available."""
        config = _make_config(vision_provider="anthropic")
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_for_capability(Capability.VISION)
        assert result is mock_provider

    def test_returns_preferred_reasoning_provider(self):
        """Preferred reasoning provider is returned when available."""
        config = _make_config(reasoning_provider="openai")
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "openai",
            frozenset({Capability.REASON}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_for_capability(Capability.REASON)
        assert result is mock_provider

    def test_returns_preferred_embedding_provider(self):
        """Preferred embedding provider is returned when available."""
        config = _make_config(embedding_provider="voyage")
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "voyage",
            frozenset({Capability.EMBED}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_for_capability(Capability.EMBED)
        assert result is mock_provider

    def test_falls_back_when_preferred_unavailable(self):
        """Fallback chain is tried when preferred provider fails."""
        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
        )
        router = ProviderRouter(config=config)

        mock_openai = _make_mock_provider(
            "openai",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        def _mock_get_provider(name, **kwargs):
            if name == "anthropic":
                raise ImportError("anthropic not installed")
            if name == "openai":
                return mock_openai
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            result = router.get_for_capability(Capability.VISION)
        assert result is mock_openai

    def test_claude_code_ultimate_fallback_vision(self):
        """Claude-code is used as ultimate fallback for VISION."""
        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai"],
        )
        router = ProviderRouter(config=config)

        mock_claude = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        def _mock_get_provider(name, **kwargs):
            if name == "claude-code":
                return mock_claude
            # All others fail
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            result = router.get_for_capability(Capability.VISION)
        assert result is mock_claude

    def test_claude_code_ultimate_fallback_reason(self):
        """Claude-code is used as ultimate fallback for REASON."""
        config = _make_config(
            reasoning_provider="openai",
            reasoning_fallbacks=["anthropic"],
        )
        router = ProviderRouter(config=config)

        mock_claude = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        def _mock_get_provider(name, **kwargs):
            if name == "claude-code":
                return mock_claude
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            result = router.get_for_capability(Capability.REASON)
        assert result is mock_claude

    def test_no_claude_code_fallback_for_transcribe(self):
        """Claude-code is NOT used as fallback for TRANSCRIBE."""
        config = _make_config(
            transcription_provider="openai",
            transcription_fallbacks=["deepgram"],
        )
        router = ProviderRouter(config=config)

        def _mock_get_provider(name, **kwargs):
            raise ImportError(f"{name} not installed")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get_provider,
            ),
            pytest.raises(NoProviderError) as exc_info,
        ):
            router.get_for_capability(Capability.TRANSCRIBE)

        assert exc_info.value.capability == Capability.TRANSCRIBE

    def test_no_provider_error_for_embed(self):
        """NoProviderError raised when no embedding provider works."""
        config = _make_config(embedding_provider="voyage")
        router = ProviderRouter(config=config)

        def _mock_get_provider(name, **kwargs):
            raise ImportError(f"{name} not installed")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get_provider,
            ),
            pytest.raises(NoProviderError) as exc_info,
        ):
            router.get_for_capability(Capability.EMBED)

        assert exc_info.value.capability == Capability.EMBED

    def test_skips_unavailable_provider(self):
        """Provider that returns is_available=False is skipped."""
        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
        )
        router = ProviderRouter(config=config)

        mock_unavailable = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION}),
            available=False,
        )
        mock_openai = _make_mock_provider(
            "openai",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        def _mock_get_provider(name, **kwargs):
            if name == "anthropic":
                return mock_unavailable
            if name == "openai":
                return mock_openai
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get_provider,
        ):
            result = router.get_for_capability(Capability.VISION)
        assert result is mock_openai

    def test_deduplicates_provider_names(self):
        """Same provider name in preferred and fallback is only tried once."""
        config = _make_config(
            vision_provider="claude-code",
            vision_fallbacks=["claude-code"],
        )
        router = ProviderRouter(config=config)

        call_count = 0

        def _mock_get_provider(name, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ImportError(f"{name} not installed")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get_provider,
            ),
            pytest.raises(NoProviderError),
        ):
            router.get_for_capability(Capability.VISION)

        # claude-code should only be tried once even though it appears
        # in preferred, fallbacks, and ultimate fallback
        assert call_count == 1


# =============================================================================
# Convenience method tests
# =============================================================================


class TestGetTranscriber:
    """Tests for ProviderRouter.get_transcriber."""

    def test_returns_transcriber(self):
        """get_transcriber returns a Transcriber-compatible provider."""
        config = _make_config()
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_transcriber()
        # The mock has a transcribe method, so isinstance check should pass
        assert hasattr(result, "transcribe")

    def test_raises_when_no_transcriber(self):
        """get_transcriber raises NoProviderError when none available."""
        config = _make_config()
        router = ProviderRouter(config=config)

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=ImportError("not installed"),
            ),
            pytest.raises(NoProviderError),
        ):
            router.get_transcriber()


class TestGetVisionAnalyzer:
    """Tests for ProviderRouter.get_vision_analyzer."""

    def test_returns_vision_analyzer(self):
        """get_vision_analyzer returns a VisionAnalyzer-compatible provider."""
        config = _make_config()
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_vision_analyzer()
        assert hasattr(result, "analyze_images")

    def test_raises_when_no_vision_analyzer(self):
        """get_vision_analyzer raises NoProviderError when none available."""
        config = _make_config(
            vision_provider="nonexistent",
            vision_fallbacks=[],
        )
        router = ProviderRouter(config=config)

        def _mock_get_provider(name, **kwargs):
            raise ImportError(f"{name} not installed")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get_provider,
            ),
            pytest.raises(NoProviderError),
        ):
            router.get_vision_analyzer()


class TestGetVideoAnalyzer:
    """Tests for ProviderRouter.get_video_analyzer."""

    def test_returns_none_when_not_configured(self):
        """get_video_analyzer returns None when no video provider set."""
        config = _make_config(video_provider=None)
        router = ProviderRouter(config=config)
        assert router.get_video_analyzer() is None

    def test_returns_video_analyzer(self):
        """get_video_analyzer returns provider when configured."""
        config = _make_config(video_provider="google")
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "google",
            frozenset({Capability.VISION, Capability.VIDEO, Capability.REASON}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_video_analyzer()
        assert result is mock_provider

    def test_returns_none_when_provider_unavailable(self):
        """get_video_analyzer returns None when provider not available."""
        config = _make_config(video_provider="google")
        router = ProviderRouter(config=config)

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=ImportError("not installed"),
        ):
            result = router.get_video_analyzer()
        assert result is None

    def test_returns_none_when_provider_not_video_capable(self):
        """get_video_analyzer returns None for non-VideoAnalyzer provider."""
        config = _make_config(video_provider="openai")
        router = ProviderRouter(config=config)

        # openai doesn't have VIDEO capability, so no analyze_video method
        mock_provider = _make_mock_provider(
            "openai",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_video_analyzer()
        assert result is None


class TestGetReasoner:
    """Tests for ProviderRouter.get_reasoner."""

    def test_returns_reasoner(self):
        """get_reasoner returns a Reasoner-compatible provider."""
        config = _make_config()
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_reasoner()
        assert hasattr(result, "reason")


class TestGetEmbedder:
    """Tests for ProviderRouter.get_embedder."""

    def test_returns_embedder(self):
        """get_embedder returns an Embedder-compatible provider."""
        config = _make_config()
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "voyage",
            frozenset({Capability.EMBED}),
        )

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = router.get_embedder()
        assert hasattr(result, "embed")

    def test_raises_when_no_embedder(self):
        """get_embedder raises NoProviderError when none available."""
        config = _make_config()
        router = ProviderRouter(config=config)

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=ImportError("not installed"),
            ),
            pytest.raises(NoProviderError),
        ):
            router.get_embedder()


# =============================================================================
# call_with_fallback tests
# =============================================================================


class TestCallWithFallback:
    """Tests for ProviderRouter.call_with_fallback."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for fallback testing."""
        mock_primary = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
        )
        mock_fallback = _make_mock_provider(
            "openai",
            frozenset({Capability.VISION, Capability.REASON}),
        )
        mock_claude = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )
        return mock_primary, mock_fallback, mock_claude

    @pytest.mark.asyncio
    async def test_calls_preferred_provider_first(self, mock_providers):
        """call_with_fallback uses preferred provider on success."""
        mock_primary, mock_fallback, mock_claude = mock_providers
        mock_primary.analyze_images = AsyncMock(return_value="primary result")

        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
        )
        router = ProviderRouter(config=config)

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                return mock_primary
            if name == "openai":
                return mock_fallback
            if name == "claude-code":
                return mock_claude
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["img.jpg"],
                prompt="describe",
            )

        assert result == "primary result"
        mock_primary.analyze_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_error(self, mock_providers):
        """call_with_fallback tries next provider on error."""
        mock_primary, mock_fallback, mock_claude = mock_providers
        mock_primary.analyze_images = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        mock_fallback.analyze_images = AsyncMock(
            return_value="fallback result"
        )

        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai", "claude-code"],
            cost_preference="quality",
        )
        router = ProviderRouter(config=config)

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                return mock_primary
            if name == "openai":
                return mock_fallback
            if name == "claude-code":
                return mock_claude
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["img.jpg"],
                prompt="describe",
            )

        assert result == "fallback result"
        mock_primary.analyze_images.assert_called_once()
        mock_fallback.analyze_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        """call_with_fallback retries on 429 before falling back."""
        mock_provider = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION}),
        )

        # First call raises rate limit, second succeeds
        rate_limit_error = Exception("429 rate limit exceeded")
        mock_provider.analyze_images = AsyncMock(
            side_effect=[rate_limit_error, "success after retry"]
        )

        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=[],
        )
        router = ProviderRouter(config=config)

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                return_value=mock_provider,
            ),
            patch(
                "claudetube.providers.router.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["img.jpg"],
                prompt="describe",
            )

        assert result == "success after retry"
        assert mock_provider.analyze_images.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_after_rate_limit_retries_exhausted(self):
        """Falls back to next provider when all rate limit retries fail."""
        mock_primary = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION}),
        )
        mock_fallback = _make_mock_provider(
            "openai",
            frozenset({Capability.VISION}),
        )

        rate_limit_error = Exception("429 too many requests")
        mock_primary.analyze_images = AsyncMock(
            side_effect=rate_limit_error,
        )
        mock_fallback.analyze_images = AsyncMock(
            return_value="fallback result",
        )

        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai"],
        )
        router = ProviderRouter(config=config)

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                return mock_primary
            if name == "openai":
                return mock_fallback
            raise ValueError(f"Unknown: {name}")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get,
            ),
            patch(
                "claudetube.providers.router.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["img.jpg"],
                prompt="describe",
            )

        assert result == "fallback result"
        # Primary should have been called 3 times (1 + 2 retries)
        assert mock_primary.analyze_images.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_when_all_providers_fail(self):
        """NoProviderError raised when all providers fail."""
        mock_provider = _make_mock_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
        )
        mock_provider.transcribe = AsyncMock(
            side_effect=RuntimeError("transcription error")
        )

        config = _make_config(
            transcription_provider="whisper-local",
            transcription_fallbacks=[],
        )
        router = ProviderRouter(config=config)

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                return_value=mock_provider,
            ),
            pytest.raises(NoProviderError) as exc_info,
        ):
            await router.call_with_fallback(
                Capability.TRANSCRIBE,
                "transcribe",
                "audio.mp3",
            )

        assert exc_info.value.capability == Capability.TRANSCRIBE
        assert "Last error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_no_providers_available(self):
        """NoProviderError raised when no providers can be loaded."""
        config = _make_config(
            embedding_provider="voyage",
        )
        router = ProviderRouter(config=config)

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=ImportError("not installed"),
            ),
            pytest.raises(NoProviderError),
        ):
            await router.call_with_fallback(
                Capability.EMBED,
                "embed",
                "some text",
            )

    @pytest.mark.asyncio
    async def test_skips_provider_without_method(self):
        """Providers missing the requested method are skipped."""
        mock_primary = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION}),
        )
        # Remove the method we're trying to call
        del mock_primary.analyze_images

        mock_fallback = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION}),
        )
        mock_fallback.analyze_images = AsyncMock(
            return_value="fallback result"
        )

        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["claude-code"],
        )
        router = ProviderRouter(config=config)

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                return mock_primary
            if name == "claude-code":
                return mock_fallback
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["img.jpg"],
                prompt="describe",
            )

        assert result == "fallback result"


# =============================================================================
# Logging tests
# =============================================================================


class TestLogging:
    """Tests for provider selection logging."""

    def test_logs_preferred_selection_at_info(self, caplog):
        """Selecting preferred provider logs at INFO level."""
        config = _make_config(transcription_provider="whisper-local")
        router = ProviderRouter(config=config)

        mock_provider = _make_mock_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
        )

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                return_value=mock_provider,
            ),
            caplog.at_level(logging.INFO, logger="claudetube.providers.router"),
        ):
            router.get_for_capability(Capability.TRANSCRIBE)

        assert "preferred provider" in caplog.text.lower()
        assert "whisper-local" in caplog.text

    def test_logs_fallback_at_warning(self, caplog):
        """Falling back to another provider logs at WARNING level."""
        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai"],
        )
        router = ProviderRouter(config=config)

        mock_openai = _make_mock_provider(
            "openai",
            frozenset({Capability.VISION}),
        )

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                raise ImportError("not installed")
            if name == "openai":
                return mock_openai
            raise ValueError(f"Unknown: {name}")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get,
            ),
            caplog.at_level(
                logging.WARNING, logger="claudetube.providers.router"
            ),
        ):
            router.get_for_capability(Capability.VISION)

        # Should log warning about preferred provider being unavailable
        assert "anthropic" in caplog.text

    def test_logs_claude_code_ultimate_fallback(self, caplog):
        """Using claude-code as ultimate fallback logs at INFO level."""
        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai"],
        )
        router = ProviderRouter(config=config)

        mock_claude = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        def _mock_get(name, **kwargs):
            if name == "claude-code":
                return mock_claude
            raise ImportError(f"{name} not installed")

        with (
            patch(
                "claudetube.providers.registry.get_provider",
                side_effect=_mock_get,
            ),
            caplog.at_level(logging.INFO, logger="claudetube.providers.router"),
        ):
            router.get_for_capability(Capability.VISION)

        assert "ultimate fallback" in caplog.text.lower()


# =============================================================================
# _is_rate_limit_error tests
# =============================================================================


class TestIsRateLimitError:
    """Tests for the _is_rate_limit_error helper."""

    def test_detects_status_code_429(self):
        """Detects error with status_code=429 attribute."""
        err = Exception("rate limited")
        err.status_code = 429  # type: ignore[attr-defined]
        assert _is_rate_limit_error(err) is True

    def test_detects_response_status_429(self):
        """Detects error with response.status_code=429."""
        err = Exception("rate limited")
        err.response = MagicMock()  # type: ignore[attr-defined]
        err.response.status_code = 429
        assert _is_rate_limit_error(err) is True

    def test_detects_429_in_message(self):
        """Detects '429' in error message string."""
        err = Exception("HTTP 429 Too Many Requests")
        assert _is_rate_limit_error(err) is True

    def test_detects_rate_limit_in_message(self):
        """Detects 'rate limit' in error message string."""
        err = Exception("Rate limit exceeded, please try again later")
        assert _is_rate_limit_error(err) is True

    def test_non_rate_limit_error(self):
        """Regular errors are not classified as rate limits."""
        err = Exception("Internal server error")
        assert _is_rate_limit_error(err) is False

    def test_non_429_status_code(self):
        """Non-429 status codes are not classified as rate limits."""
        err = Exception("server error")
        err.status_code = 500  # type: ignore[attr-defined]
        assert _is_rate_limit_error(err) is False


# =============================================================================
# Integration-style tests (still mocked, but test full flows)
# =============================================================================


class TestFullRoutingFlow:
    """Integration tests for complete routing flows."""

    def test_transcription_flow_no_claude_code_fallback(self):
        """Full transcription routing never uses claude-code."""
        config = _make_config(
            transcription_provider="openai",
            transcription_fallbacks=["deepgram", "whisper-local"],
            cost_preference="quality",
        )
        router = ProviderRouter(config=config)

        mock_whisper = _make_mock_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
        )

        attempted_providers = []

        def _mock_get(name, **kwargs):
            attempted_providers.append(name)
            if name == "whisper-local":
                return mock_whisper
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = router.get_for_capability(Capability.TRANSCRIBE)

        assert result is mock_whisper
        # Should have tried openai, deepgram, then whisper-local
        assert attempted_providers == ["openai", "deepgram", "whisper-local"]
        # Should NOT have tried claude-code
        assert "claude-code" not in attempted_providers

    def test_vision_flow_with_ultimate_fallback(self):
        """Full vision routing falls through to claude-code."""
        config = _make_config(
            vision_provider="anthropic",
            vision_fallbacks=["openai"],
        )
        router = ProviderRouter(config=config)

        mock_claude = _make_mock_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        attempted_providers = []

        def _mock_get(name, **kwargs):
            attempted_providers.append(name)
            if name == "claude-code":
                return mock_claude
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = router.get_for_capability(Capability.VISION)

        assert result is mock_claude
        assert attempted_providers == ["anthropic", "openai", "claude-code"]

    @pytest.mark.asyncio
    async def test_call_with_fallback_complete_chain(self):
        """call_with_fallback tries entire chain before raising."""
        mock_p1 = _make_mock_provider(
            "anthropic", frozenset({Capability.REASON})
        )
        mock_p2 = _make_mock_provider(
            "openai", frozenset({Capability.REASON})
        )
        mock_p3 = _make_mock_provider(
            "claude-code", frozenset({Capability.REASON})
        )

        mock_p1.reason = AsyncMock(side_effect=RuntimeError("p1 error"))
        mock_p2.reason = AsyncMock(side_effect=RuntimeError("p2 error"))
        mock_p3.reason = AsyncMock(return_value="claude-code result")

        config = _make_config(
            reasoning_provider="anthropic",
            reasoning_fallbacks=["openai", "claude-code"],
            cost_preference="quality",
        )
        router = ProviderRouter(config=config)

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                return mock_p1
            if name == "openai":
                return mock_p2
            if name == "claude-code":
                return mock_p3
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = await router.call_with_fallback(
                Capability.REASON,
                "reason",
                [{"role": "user", "content": "hello"}],
            )

        assert result == "claude-code result"
        mock_p1.reason.assert_called_once()
        mock_p2.reason.assert_called_once()
        mock_p3.reason.assert_called_once()


# =============================================================================
# Cost-based routing tests
# =============================================================================


class TestCostBasedRouting:
    """Tests for cost-based provider selection."""

    def test_fallback_sorted_by_cost_when_cost_preference(self):
        """Fallback chain is sorted by cost tier when cost_preference='cost'."""
        config = _make_config(
            vision_provider="nonexistent",
            vision_fallbacks=["anthropic", "google", "openai"],
            cost_preference="cost",
        )
        router = ProviderRouter(config=config)

        # anthropic=EXPENSIVE, google=CHEAP, openai=MODERATE
        mock_google = _make_mock_provider(
            "google",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        attempted_providers = []

        def _mock_get(name, **kwargs):
            attempted_providers.append(name)
            if name == "google":
                return mock_google
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = router.get_for_capability(Capability.VISION)

        assert result is mock_google
        # nonexistent tried first (preferred), then fallbacks sorted by cost:
        # google (CHEAP) before openai (MODERATE) before anthropic (EXPENSIVE)
        assert attempted_providers[0] == "nonexistent"
        # After preferred fails, fallbacks should be cost-sorted
        fallback_attempts = attempted_providers[1:]
        assert fallback_attempts[0] == "google"

    def test_fallback_not_sorted_when_quality_preference(self):
        """Fallback chain preserves config order when cost_preference='quality'."""
        config = _make_config(
            vision_provider="nonexistent",
            vision_fallbacks=["anthropic", "google", "openai"],
            cost_preference="quality",
        )
        router = ProviderRouter(config=config)

        mock_anthropic = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        attempted_providers = []

        def _mock_get(name, **kwargs):
            attempted_providers.append(name)
            if name == "anthropic":
                return mock_anthropic
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = router.get_for_capability(Capability.VISION)

        assert result is mock_anthropic
        # Fallbacks in config order, not cost-sorted
        assert attempted_providers[1] == "anthropic"

    def test_preferred_provider_always_first_regardless_of_cost(self):
        """Preferred provider is tried first even if it's expensive."""
        config = _make_config(
            vision_provider="anthropic",  # EXPENSIVE
            vision_fallbacks=["google"],  # CHEAP
            cost_preference="cost",
        )
        router = ProviderRouter(config=config)

        mock_anthropic = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
        )

        def _mock_get(name, **kwargs):
            if name == "anthropic":
                return mock_anthropic
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = router.get_for_capability(Capability.VISION)

        # Even though anthropic is EXPENSIVE, it's the preferred provider
        assert result is mock_anthropic

    def test_sort_by_cost_method(self):
        """_sort_by_cost correctly orders provider names."""
        config = _make_config()
        router = ProviderRouter(config=config)

        names = ["anthropic", "whisper-local", "openai", "google", "claude-code"]
        sorted_names = router._sort_by_cost(names)

        # FREE: whisper-local, claude-code
        # CHEAP: google
        # MODERATE: openai
        # EXPENSIVE: anthropic
        assert sorted_names.index("whisper-local") < sorted_names.index("google")
        assert sorted_names.index("claude-code") < sorted_names.index("google")
        assert sorted_names.index("google") < sorted_names.index("openai")
        assert sorted_names.index("openai") < sorted_names.index("anthropic")

    def test_sort_by_cost_unknown_provider_gets_moderate(self):
        """Unknown providers default to MODERATE cost tier."""
        config = _make_config()
        router = ProviderRouter(config=config)

        names = ["unknown-provider", "whisper-local", "anthropic"]
        sorted_names = router._sort_by_cost(names)

        # whisper-local (FREE) < unknown (MODERATE) < anthropic (EXPENSIVE)
        assert sorted_names.index("whisper-local") < sorted_names.index("unknown-provider")
        assert sorted_names.index("unknown-provider") < sorted_names.index("anthropic")

    def test_sort_by_cost_stable_within_same_tier(self):
        """Providers with same cost tier preserve original order."""
        config = _make_config()
        router = ProviderRouter(config=config)

        # whisper-local, claude-code, ollama are all FREE
        names = ["ollama", "claude-code", "whisper-local"]
        sorted_names = router._sort_by_cost(names)

        # All FREE, so original order preserved
        assert sorted_names == ["ollama", "claude-code", "whisper-local"]

    @pytest.mark.asyncio
    async def test_call_with_fallback_cost_sorted(self):
        """call_with_fallback respects cost-based ordering in fallback chain."""
        mock_google = _make_mock_provider(
            "google",
            frozenset({Capability.VISION, Capability.REASON}),
        )
        mock_google.analyze_images = AsyncMock(return_value="google result")

        mock_anthropic = _make_mock_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
        )
        mock_anthropic.analyze_images = AsyncMock(return_value="anthropic result")

        config = _make_config(
            vision_provider="nonexistent",
            vision_fallbacks=["anthropic", "google"],
            cost_preference="cost",
        )
        router = ProviderRouter(config=config)

        def _mock_get(name, **kwargs):
            if name == "google":
                return mock_google
            if name == "anthropic":
                return mock_anthropic
            raise ImportError(f"{name} not installed")

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=_mock_get,
        ):
            result = await router.call_with_fallback(
                Capability.VISION,
                "analyze_images",
                ["img.jpg"],
                prompt="describe",
            )

        # google (CHEAP) should be tried before anthropic (EXPENSIVE)
        assert result == "google result"
        mock_google.analyze_images.assert_called_once()
        mock_anthropic.analyze_images.assert_not_called()

    def test_default_cost_preference_is_cost(self):
        """Default cost_preference should be 'cost'."""
        config = ProvidersConfig()
        assert config.cost_preference == "cost"


# =============================================================================
# Module exports test
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exported(self):
        """All expected items are in __all__."""
        from claudetube.providers import router

        assert "ProviderRouter" in router.__all__
        assert "NoProviderError" in router.__all__

    def test_can_import_router(self):
        """ProviderRouter can be imported from router module."""
        from claudetube.providers.router import ProviderRouter

        assert ProviderRouter is not None

    def test_can_import_error(self):
        """NoProviderError can be imported from router module."""
        from claudetube.providers.router import NoProviderError

        assert NoProviderError is not None
