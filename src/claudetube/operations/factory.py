"""
Operation factory for constructing operations with configured providers.

Provides a single entry point for creating operation instances with the
appropriate AI providers based on configuration preferences and fallbacks.
The factory resolves provider names from ProvidersConfig, instantiates them
via the registry, and injects them into operation constructors.

Classes:
    OperationFactory: Constructs operation instances with configured providers.

Example:
    >>> from claudetube.providers.config import get_providers_config
    >>> factory = OperationFactory(get_providers_config())
    >>> op = factory.get_transcribe_operation()
    >>> result = await op.execute(video_id, audio_path)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claudetube.providers.base import (
    Reasoner,
    Transcriber,
    VideoAnalyzer,
    VisionAnalyzer,
)
from claudetube.providers.capabilities import PROVIDER_INFO, Capability
from claudetube.providers.config import ProvidersConfig, get_providers_config
from claudetube.providers.registry import get_provider

if TYPE_CHECKING:
    from claudetube.providers.base import Provider

logger = logging.getLogger(__name__)


class OperationFactory:
    """Construct operation instances with configured providers.

    Uses ProvidersConfig preferences and fallback chains to select
    providers, then injects them into operation constructors. Caches
    resolved providers to avoid repeated lookups.

    Args:
        config: Provider configuration. If None, loads from global config.

    Example:
        >>> factory = OperationFactory()
        >>> transcribe_op = factory.get_transcribe_operation()
        >>> visual_op = factory.get_visual_operation()
    """

    def __init__(self, config: ProvidersConfig | None = None):
        self.config = config or get_providers_config()
        self._provider_cache: dict[str, Provider] = {}

    def _resolve_provider(
        self,
        capability: Capability,
        preferred: str | None,
        fallbacks: list[str],
        **kwargs,
    ) -> Provider | None:
        """Resolve a provider for a capability using preference + fallbacks.

        Tries the preferred provider first, then each fallback in order.
        Skips providers that don't support the capability, can't be imported,
        or aren't available.

        Args:
            capability: Required capability.
            preferred: Preferred provider name (from config).
            fallbacks: Fallback chain (from config).
            **kwargs: Extra kwargs passed to get_provider().

        Returns:
            A Provider instance, or None if no provider could be resolved.
        """
        candidates = []
        if preferred:
            candidates.append(preferred)
        for fb in fallbacks:
            if fb not in candidates:
                candidates.append(fb)

        for name in candidates:
            # Check capability from static info first (avoids import)
            info = PROVIDER_INFO.get(name)
            if info and not info.can(capability):
                logger.debug(
                    "Skipping '%s' for %s: lacks capability",
                    name,
                    capability.name,
                )
                continue

            try:
                provider = self._get_or_create_provider(name, **kwargs)
            except (ImportError, ValueError, TypeError) as e:
                logger.debug("Skipping '%s' for %s: %s", name, capability.name, e)
                continue

            if not provider.is_available():
                logger.debug(
                    "Skipping '%s' for %s: not available",
                    name,
                    capability.name,
                )
                continue

            logger.debug("Resolved '%s' for %s", name, capability.name)
            return provider

        logger.warning("No provider found for %s", capability.name)
        return None

    def _get_or_create_provider(self, name: str, **kwargs) -> Provider:
        """Get a cached provider or create a new one.

        Args:
            name: Canonical provider name.
            **kwargs: Extra kwargs for provider construction.

        Returns:
            Provider instance.
        """
        cache_key = name if not kwargs else None
        if cache_key and cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        provider = get_provider(name, **kwargs)

        if cache_key:
            self._provider_cache[cache_key] = provider

        return provider

    def get_transcriber(self, **kwargs) -> Transcriber | None:
        """Resolve a Transcriber provider.

        Args:
            **kwargs: Extra kwargs for provider construction
                (e.g., model_size for whisper-local).

        Returns:
            A provider implementing Transcriber, or None.
        """
        provider = self._resolve_provider(
            Capability.TRANSCRIBE,
            self.config.transcription_provider,
            self.config.transcription_fallbacks,
            **kwargs,
        )
        if provider and isinstance(provider, Transcriber):
            return provider
        return None

    def get_vision_analyzer(self) -> VisionAnalyzer | None:
        """Resolve a VisionAnalyzer provider.

        Returns:
            A provider implementing VisionAnalyzer, or None.
        """
        provider = self._resolve_provider(
            Capability.VISION,
            self.config.vision_provider,
            self.config.vision_fallbacks,
        )
        if provider and isinstance(provider, VisionAnalyzer):
            return provider
        return None

    def get_video_analyzer(self) -> VideoAnalyzer | None:
        """Resolve a VideoAnalyzer provider.

        Returns:
            A provider implementing VideoAnalyzer, or None.
        """
        provider = self._resolve_provider(
            Capability.VIDEO,
            self.config.video_provider,
            [],  # No fallback chain for video (only google supports it)
        )
        if provider and isinstance(provider, VideoAnalyzer):
            return provider
        return None

    def get_reasoner(self) -> Reasoner | None:
        """Resolve a Reasoner provider.

        Returns:
            A provider implementing Reasoner, or None.
        """
        provider = self._resolve_provider(
            Capability.REASON,
            self.config.reasoning_provider,
            self.config.reasoning_fallbacks,
        )
        if provider and isinstance(provider, Reasoner):
            return provider
        return None

    def get_transcribe_operation(self, **kwargs):
        """Create a TranscribeOperation with configured provider.

        Args:
            **kwargs: Extra kwargs for the transcriber provider
                (e.g., model_size="small" for whisper-local).

        Returns:
            TranscribeOperation ready to execute.

        Raises:
            RuntimeError: If no transcriber provider is available.
        """
        from claudetube.operations.transcribe import TranscribeOperation

        transcriber = self.get_transcriber(**kwargs)
        if transcriber is None:
            raise RuntimeError(
                "No transcription provider available. "
                "Check your provider configuration and installed dependencies."
            )
        return TranscribeOperation(transcriber)

    def get_visual_operation(self):
        """Create a VisualTranscriptOperation with configured provider.

        Returns:
            VisualTranscriptOperation ready to execute.

        Raises:
            RuntimeError: If no vision provider is available.
        """
        from claudetube.operations.visual_transcript import (
            VisualTranscriptOperation,
        )

        vision = self.get_vision_analyzer()
        if vision is None:
            raise RuntimeError(
                "No vision provider available. "
                "Check your provider configuration and installed dependencies."
            )
        return VisualTranscriptOperation(vision)

    def get_entity_extraction_operation(self):
        """Create an EntityExtractionOperation with configured providers.

        Uses VideoAnalyzer (most efficient, native video), VisionAnalyzer
        (frame-by-frame fallback), and Reasoner for semantic concept extraction.
        Any may be None if unavailable; the operation handles partial
        availability gracefully.

        Returns:
            EntityExtractionOperation ready to execute.
        """
        from claudetube.operations.entity_extraction import (
            EntityExtractionOperation,
        )

        video_analyzer = self.get_video_analyzer()
        vision = self.get_vision_analyzer()
        reasoner = self.get_reasoner()
        return EntityExtractionOperation(
            video_analyzer=video_analyzer,
            vision_analyzer=vision,
            reasoner=reasoner,
        )

    def get_person_tracking_operation(self):
        """Create a PersonTrackingOperation with configured providers.

        Uses VideoAnalyzer (Gemini, most efficient) and/or VisionAnalyzer
        for frame-by-frame analysis. The operation implements its own
        tier priority when both are available.

        Returns:
            PersonTrackingOperation ready to execute.
        """
        from claudetube.operations.person_tracking import (
            PersonTrackingOperation,
        )

        video_analyzer = self.get_video_analyzer()
        vision = self.get_vision_analyzer()
        return PersonTrackingOperation(
            video_analyzer=video_analyzer, vision_analyzer=vision
        )

    def clear_cache(self) -> None:
        """Clear the factory's internal provider cache."""
        self._provider_cache.clear()


# Module-level singleton for convenience
_factory: OperationFactory | None = None


def get_factory(config: ProvidersConfig | None = None) -> OperationFactory:
    """Get the global OperationFactory singleton.

    Creates on first call, then returns cached. Pass config to override
    the default configuration (useful for testing).

    Args:
        config: Optional ProvidersConfig override.

    Returns:
        OperationFactory instance.
    """
    global _factory
    if _factory is None or config is not None:
        _factory = OperationFactory(config)
    return _factory


def clear_factory_cache() -> None:
    """Clear the global factory singleton (for testing or config reload)."""
    global _factory
    _factory = None


__all__ = [
    "OperationFactory",
    "get_factory",
    "clear_factory_cache",
]
