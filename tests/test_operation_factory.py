"""Tests for OperationFactory.

Verifies:
1. Factory construction with config
2. Provider resolution with preferences and fallbacks
3. Capability checking skips unsuitable providers
4. Unavailable providers are skipped
5. Operation construction for all four operation types
6. Factory caches resolved providers
7. get_factory() singleton behavior
8. Error cases (no provider available)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claudetube.operations.factory import (
    OperationFactory,
    clear_factory_cache,
    get_factory,
)
from claudetube.providers.base import (
    Reasoner,
    Transcriber,
    VideoAnalyzer,
    VisionAnalyzer,
)
from claudetube.providers.capabilities import Capability, ProviderInfo
from claudetube.providers.config import ProvidersConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    name: str,
    capabilities: frozenset[Capability],
    available: bool = True,
    protocols: tuple = (),
):
    """Create a mock provider that satisfies given protocols.

    Uses a dynamic class that inherits from the requested protocol classes
    so isinstance() checks pass.
    """
    bases = (*protocols, object) if protocols else (object,)

    # Build a class dynamically so isinstance checks work
    class MockProvider(*bases):
        pass

    provider = MockProvider()
    info = ProviderInfo(name=name, capabilities=capabilities)
    provider.info = info
    provider.is_available = MagicMock(return_value=available)

    # Add async method stubs matching protocols
    if Transcriber in protocols:
        provider.transcribe = MagicMock()
    if VisionAnalyzer in protocols:
        provider.analyze_images = MagicMock()
    if VideoAnalyzer in protocols:
        provider.analyze_video = MagicMock()
    if Reasoner in protocols:
        provider.reason = MagicMock()

    return provider


def _default_config(**overrides) -> ProvidersConfig:
    """Create a ProvidersConfig with sensible test defaults."""
    defaults = {
        "transcription_provider": "whisper-local",
        "vision_provider": "anthropic",
        "video_provider": "google",
        "reasoning_provider": "anthropic",
        "transcription_fallbacks": ["whisper-local"],
        "vision_fallbacks": ["anthropic", "claude-code"],
        "reasoning_fallbacks": ["anthropic", "claude-code"],
    }
    defaults.update(overrides)
    return ProvidersConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests: Factory construction
# ---------------------------------------------------------------------------


class TestFactoryConstruction:
    def test_accepts_config(self):
        config = _default_config()
        factory = OperationFactory(config)
        assert factory.config is config

    @patch("claudetube.operations.factory.get_providers_config")
    def test_defaults_to_global_config(self, mock_get_config):
        mock_config = _default_config()
        mock_get_config.return_value = mock_config
        factory = OperationFactory()
        assert factory.config is mock_config


# ---------------------------------------------------------------------------
# Tests: Provider resolution
# ---------------------------------------------------------------------------


class TestProviderResolution:
    def test_resolves_preferred_provider(self):
        config = _default_config(transcription_provider="whisper-local")
        factory = OperationFactory(config)

        mock_provider = _make_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
            protocols=(Transcriber,),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=mock_provider,
        ):
            result = factory.get_transcriber()

        assert result is mock_provider

    def test_falls_back_when_preferred_unavailable(self):
        config = _default_config(
            vision_provider="anthropic",
            vision_fallbacks=["anthropic", "claude-code"],
        )
        factory = OperationFactory(config)

        anthropic_provider = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            available=False,
            protocols=(VisionAnalyzer,),
        )
        claude_code_provider = _make_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
            available=True,
            protocols=(VisionAnalyzer,),
        )

        def mock_get_provider(name, **kwargs):
            if name == "anthropic":
                return anthropic_provider
            if name == "claude-code":
                return claude_code_provider
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.operations.factory.get_provider",
            side_effect=mock_get_provider,
        ):
            result = factory.get_vision_analyzer()

        assert result is claude_code_provider

    def test_skips_provider_lacking_capability(self):
        """A provider that doesn't support the needed capability is skipped."""
        config = _default_config(
            transcription_provider="anthropic",  # anthropic can't transcribe
            transcription_fallbacks=["whisper-local"],
        )
        factory = OperationFactory(config)

        whisper_provider = _make_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
            protocols=(Transcriber,),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=whisper_provider,
        ):
            result = factory.get_transcriber()

        # anthropic should be skipped (no TRANSCRIBE capability in PROVIDER_INFO)
        # whisper-local should be returned
        assert result is whisper_provider

    def test_skips_provider_with_import_error(self):
        config = _default_config(
            vision_provider="anthropic",
            vision_fallbacks=["anthropic", "claude-code"],
        )
        factory = OperationFactory(config)

        claude_code_provider = _make_provider(
            "claude-code",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer,),
        )

        def mock_get_provider(name, **kwargs):
            if name == "anthropic":
                raise ImportError("anthropic not installed")
            return claude_code_provider

        with patch(
            "claudetube.operations.factory.get_provider",
            side_effect=mock_get_provider,
        ):
            result = factory.get_vision_analyzer()

        assert result is claude_code_provider

    def test_returns_none_when_no_provider_available(self):
        config = _default_config(
            transcription_provider="whisper-local",
            transcription_fallbacks=["whisper-local"],
        )
        factory = OperationFactory(config)

        with patch(
            "claudetube.operations.factory.get_provider",
            side_effect=ImportError("not installed"),
        ):
            result = factory.get_transcriber()

        assert result is None

    def test_deduplicates_candidates(self):
        """If preferred is also in fallbacks, it's only tried once."""
        config = _default_config(
            vision_provider="anthropic",
            vision_fallbacks=["anthropic", "claude-code"],
        )
        factory = OperationFactory(config)

        call_count = {"anthropic": 0}

        def mock_get_provider(name, **kwargs):
            if name == "anthropic":
                call_count["anthropic"] += 1
                raise ImportError("not installed")
            raise ImportError("not installed")

        with patch(
            "claudetube.operations.factory.get_provider",
            side_effect=mock_get_provider,
        ):
            factory.get_vision_analyzer()

        # anthropic should only be tried once despite appearing
        # in both preferred and fallbacks
        assert call_count["anthropic"] == 1


# ---------------------------------------------------------------------------
# Tests: Provider caching
# ---------------------------------------------------------------------------


class TestProviderCaching:
    def test_caches_resolved_providers(self):
        config = _default_config()
        factory = OperationFactory(config)

        mock_provider = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer,),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=mock_provider,
        ) as mock_get:
            result1 = factory.get_vision_analyzer()
            result2 = factory.get_vision_analyzer()

        assert result1 is result2
        # get_provider should only be called once due to caching
        assert mock_get.call_count == 1

    def test_clear_cache(self):
        config = _default_config()
        factory = OperationFactory(config)

        mock_provider = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer,),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=mock_provider,
        ) as mock_get:
            factory.get_vision_analyzer()
            factory.clear_cache()
            factory.get_vision_analyzer()

        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Operation construction
# ---------------------------------------------------------------------------


class TestOperationConstruction:
    def test_get_transcribe_operation(self):
        from claudetube.operations.transcribe import TranscribeOperation

        config = _default_config()
        factory = OperationFactory(config)

        mock_provider = _make_provider(
            "whisper-local",
            frozenset({Capability.TRANSCRIBE}),
            protocols=(Transcriber,),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=mock_provider,
        ):
            op = factory.get_transcribe_operation()

        assert isinstance(op, TranscribeOperation)
        assert op.transcriber is mock_provider

    def test_get_transcribe_operation_raises_when_none(self):
        config = _default_config()
        factory = OperationFactory(config)

        with (
            patch(
                "claudetube.operations.factory.get_provider",
                side_effect=ImportError("not installed"),
            ),
            pytest.raises(RuntimeError, match="No transcription provider"),
        ):
            factory.get_transcribe_operation()

    def test_get_visual_operation(self):
        from claudetube.operations.visual_transcript import (
            VisualTranscriptOperation,
        )

        config = _default_config()
        factory = OperationFactory(config)

        mock_provider = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer,),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=mock_provider,
        ):
            op = factory.get_visual_operation()

        assert isinstance(op, VisualTranscriptOperation)

    def test_get_visual_operation_raises_when_none(self):
        config = _default_config()
        factory = OperationFactory(config)

        with (
            patch(
                "claudetube.operations.factory.get_provider",
                side_effect=ImportError("not installed"),
            ),
            pytest.raises(RuntimeError, match="No vision provider"),
        ):
            factory.get_visual_operation()

    def test_get_entity_extraction_operation(self):
        from claudetube.operations.entity_extraction import (
            EntityExtractionOperation,
        )

        config = _default_config()
        factory = OperationFactory(config)

        mock_vision = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer, Reasoner),
        )

        with patch(
            "claudetube.operations.factory.get_provider",
            return_value=mock_vision,
        ):
            op = factory.get_entity_extraction_operation()

        assert isinstance(op, EntityExtractionOperation)
        assert op.vision is mock_vision
        assert op.reasoner is mock_vision

    def test_get_entity_extraction_partial_providers(self):
        """EntityExtractionOperation works with only vision or only reasoner."""
        from claudetube.operations.entity_extraction import (
            EntityExtractionOperation,
        )

        config = _default_config(
            reasoning_provider="nonexistent",
            reasoning_fallbacks=[],
        )
        factory = OperationFactory(config)

        mock_vision = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer,),
        )

        def mock_get_provider(name, **kwargs):
            if name == "anthropic":
                return mock_vision
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.operations.factory.get_provider",
            side_effect=mock_get_provider,
        ):
            op = factory.get_entity_extraction_operation()

        assert isinstance(op, EntityExtractionOperation)
        assert op.vision is mock_vision
        # Reasoner is None because "nonexistent" doesn't resolve
        assert op.reasoner is None

    def test_get_person_tracking_operation(self):
        from claudetube.operations.person_tracking import (
            PersonTrackingOperation,
        )

        config = _default_config()
        factory = OperationFactory(config)

        mock_google = _make_provider(
            "google",
            frozenset({Capability.VISION, Capability.VIDEO, Capability.REASON}),
            protocols=(VideoAnalyzer, VisionAnalyzer),
        )
        mock_anthropic = _make_provider(
            "anthropic",
            frozenset({Capability.VISION, Capability.REASON}),
            protocols=(VisionAnalyzer,),
        )

        def mock_get_provider(name, **kwargs):
            if name == "google":
                return mock_google
            if name == "anthropic":
                return mock_anthropic
            raise ValueError(f"Unknown: {name}")

        with patch(
            "claudetube.operations.factory.get_provider",
            side_effect=mock_get_provider,
        ):
            op = factory.get_person_tracking_operation()

        assert isinstance(op, PersonTrackingOperation)
        assert op.video_analyzer is mock_google
        assert op.vision is mock_anthropic


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def teardown_method(self):
        clear_factory_cache()

    @patch("claudetube.operations.factory.get_providers_config")
    def test_get_factory_returns_singleton(self, mock_get_config):
        mock_get_config.return_value = _default_config()
        f1 = get_factory()
        f2 = get_factory()
        assert f1 is f2

    def test_get_factory_with_config_replaces_singleton(self):
        config1 = _default_config(vision_provider="anthropic")
        config2 = _default_config(vision_provider="claude-code")

        f1 = get_factory(config1)
        f2 = get_factory(config2)

        assert f1 is not f2
        assert f2.config.vision_provider == "claude-code"

    def test_clear_factory_cache(self):
        config = _default_config()
        f1 = get_factory(config)
        clear_factory_cache()
        f2 = get_factory(config)
        assert f1 is not f2
