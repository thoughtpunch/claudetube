"""Tests for provider base classes and protocols.

Verifies that:
1. Provider ABC works correctly
2. All protocols are runtime_checkable and work with isinstance()
3. Mock implementations satisfy protocol contracts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from claudetube.providers.base import (
    Embedder,
    Provider,
    Reasoner,
    Transcriber,
    VideoAnalyzer,
    VisionAnalyzer,
)

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.capabilities import ProviderInfo


class TestProviderABC:
    """Tests for the Provider abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Provider ABC cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError, match="abstract"):
            Provider()  # type: ignore

    def test_subclass_must_implement_info(self):
        """Subclass must implement info property."""

        class IncompleteProvider(Provider):
            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    def test_subclass_must_implement_is_available(self):
        """Subclass must implement is_available method."""

        class IncompleteProvider(Provider):
            @property
            def info(self):
                return None

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    def test_complete_subclass_instantiates(self):
        """A complete subclass can be instantiated."""
        try:
            from claudetube.providers.capabilities import Capability, ProviderInfo
        except ImportError:
            pytest.skip("Capability/ProviderInfo not yet implemented (claudetube-p5o)")

        class CompleteProvider(Provider):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(
                    name="test-provider",
                    capabilities=frozenset({Capability.REASON}),
                )

            def is_available(self) -> bool:
                return True

        provider = CompleteProvider()
        assert provider.is_available() is True
        assert provider.info.name == "test-provider"


class TestTranscriberProtocol:
    """Tests for the Transcriber protocol."""

    def test_is_runtime_checkable(self):
        """Transcriber protocol can be used with isinstance()."""
        # Protocol should be decorated with @runtime_checkable
        assert getattr(Transcriber, "_is_runtime_protocol", False)

    def test_isinstance_with_compliant_class(self):
        """A class implementing transcribe() satisfies the protocol."""

        class MockTranscriber:
            async def transcribe(
                self,
                audio: Path,
                language: str | None = None,
                **kwargs,
            ):
                pass

        assert isinstance(MockTranscriber(), Transcriber)

    def test_isinstance_with_non_compliant_class(self):
        """A class without transcribe() does not satisfy the protocol."""

        class NotATranscriber:
            async def something_else(self):
                pass

        assert not isinstance(NotATranscriber(), Transcriber)

    def test_isinstance_with_wrong_signature(self):
        """A class with wrong signature still passes isinstance due to duck typing."""
        # Note: Protocol only checks method names exist, not signatures

        class WrongSignature:
            def transcribe(self):  # Missing required params, but name matches
                pass

        # This will pass isinstance but fail at runtime
        # That's expected behavior for runtime_checkable protocols
        assert isinstance(WrongSignature(), Transcriber)


class TestVisionAnalyzerProtocol:
    """Tests for the VisionAnalyzer protocol."""

    def test_is_runtime_checkable(self):
        """VisionAnalyzer protocol can be used with isinstance()."""
        assert getattr(VisionAnalyzer, "_is_runtime_protocol", False)

    def test_isinstance_with_compliant_class(self):
        """A class implementing analyze_images() satisfies the protocol."""

        class MockVisionAnalyzer:
            async def analyze_images(
                self,
                images: list[Path],
                prompt: str,
                schema: type | None = None,
                **kwargs,
            ) -> str | dict:
                return "analysis result"

        assert isinstance(MockVisionAnalyzer(), VisionAnalyzer)

    def test_isinstance_with_non_compliant_class(self):
        """A class without analyze_images() does not satisfy the protocol."""

        class NotAVisionAnalyzer:
            async def analyze_frames(self):  # Wrong method name
                pass

        assert not isinstance(NotAVisionAnalyzer(), VisionAnalyzer)


class TestVideoAnalyzerProtocol:
    """Tests for the VideoAnalyzer protocol."""

    def test_is_runtime_checkable(self):
        """VideoAnalyzer protocol can be used with isinstance()."""
        assert getattr(VideoAnalyzer, "_is_runtime_protocol", False)

    def test_isinstance_with_compliant_class(self):
        """A class implementing analyze_video() satisfies the protocol."""

        class MockVideoAnalyzer:
            async def analyze_video(
                self,
                video: Path,
                prompt: str,
                schema: type | None = None,
                start_time: float | None = None,
                end_time: float | None = None,
                **kwargs,
            ) -> str | dict:
                return "video analysis"

        assert isinstance(MockVideoAnalyzer(), VideoAnalyzer)

    def test_isinstance_with_non_compliant_class(self):
        """A class without analyze_video() does not satisfy the protocol."""

        class NotAVideoAnalyzer:
            async def process_video(self):  # Wrong method name
                pass

        assert not isinstance(NotAVideoAnalyzer(), VideoAnalyzer)


class TestReasonerProtocol:
    """Tests for the Reasoner protocol."""

    def test_is_runtime_checkable(self):
        """Reasoner protocol can be used with isinstance()."""
        assert getattr(Reasoner, "_is_runtime_protocol", False)

    def test_isinstance_with_compliant_class(self):
        """A class implementing reason() satisfies the protocol."""

        class MockReasoner:
            async def reason(
                self,
                messages: list[dict],
                schema: type | None = None,
                **kwargs,
            ) -> str | dict:
                return "reasoning result"

        assert isinstance(MockReasoner(), Reasoner)

    def test_isinstance_with_non_compliant_class(self):
        """A class without reason() does not satisfy the protocol."""

        class NotAReasoner:
            async def chat(self):  # Wrong method name
                pass

        assert not isinstance(NotAReasoner(), Reasoner)


class TestEmbedderProtocol:
    """Tests for the Embedder protocol."""

    def test_is_runtime_checkable(self):
        """Embedder protocol can be used with isinstance()."""
        assert getattr(Embedder, "_is_runtime_protocol", False)

    def test_isinstance_with_compliant_class(self):
        """A class implementing embed() satisfies the protocol."""

        class MockEmbedder:
            async def embed(
                self,
                text: str,
                images: list[Path] | None = None,
                **kwargs,
            ) -> list[float]:
                return [0.1, 0.2, 0.3]

        assert isinstance(MockEmbedder(), Embedder)

    def test_isinstance_with_non_compliant_class(self):
        """A class without embed() does not satisfy the protocol."""

        class NotAnEmbedder:
            async def encode(self):  # Wrong method name
                pass

        assert not isinstance(NotAnEmbedder(), Embedder)


class TestMultiProtocolProvider:
    """Tests for providers implementing multiple protocols."""

    def test_provider_with_multiple_protocols(self):
        """A provider can implement multiple capability protocols."""
        try:
            from claudetube.providers.capabilities import Capability, ProviderInfo
        except ImportError:
            pytest.skip("Capability/ProviderInfo not yet implemented (claudetube-p5o)")

        class MultiCapabilityProvider(Provider, VisionAnalyzer, Reasoner):
            @property
            def info(self) -> ProviderInfo:
                return ProviderInfo(
                    name="multi-provider",
                    capabilities=frozenset({Capability.VISION, Capability.REASON}),
                )

            def is_available(self) -> bool:
                return True

            async def analyze_images(
                self,
                images: list[Path],
                prompt: str,
                schema: type | None = None,
                **kwargs,
            ) -> str | dict:
                return "vision result"

            async def reason(
                self,
                messages: list[dict],
                schema: type | None = None,
                **kwargs,
            ) -> str | dict:
                return "reasoning result"

        provider = MultiCapabilityProvider()

        # Check base class
        assert isinstance(provider, Provider)
        assert provider.is_available() is True

        # Check protocols
        assert isinstance(provider, VisionAnalyzer)
        assert isinstance(provider, Reasoner)
        assert not isinstance(provider, Transcriber)
        assert not isinstance(provider, Embedder)

        # Check capabilities match
        assert Capability.VISION in provider.info.capabilities
        assert Capability.REASON in provider.info.capabilities


class TestProtocolExports:
    """Tests for module exports."""

    def test_all_protocols_exported_from_base(self):
        """All protocols are in base module's __all__."""
        from claudetube.providers import base

        assert "Provider" in base.__all__
        assert "Transcriber" in base.__all__
        assert "VisionAnalyzer" in base.__all__
        assert "VideoAnalyzer" in base.__all__
        assert "Reasoner" in base.__all__
        assert "Embedder" in base.__all__

    def test_all_protocols_exported_from_package(self):
        """All protocols are exported from providers package."""
        from claudetube import providers

        assert hasattr(providers, "Provider")
        assert hasattr(providers, "Transcriber")
        assert hasattr(providers, "VisionAnalyzer")
        assert hasattr(providers, "VideoAnalyzer")
        assert hasattr(providers, "Reasoner")
        assert hasattr(providers, "Embedder")

    def test_import_from_package(self):
        """Can import protocols directly from package."""
        from claudetube.providers import (
            Embedder,
            Provider,
            Reasoner,
            Transcriber,
            VideoAnalyzer,
            VisionAnalyzer,
        )

        # Just verify imports work
        assert Provider is not None
        assert Transcriber is not None
        assert VisionAnalyzer is not None
        assert VideoAnalyzer is not None
        assert Reasoner is not None
        assert Embedder is not None
