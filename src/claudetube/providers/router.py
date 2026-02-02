"""
claudetube.providers.router - Smart provider routing with fallback chains.

This module implements intelligent routing of requests to providers
based on capabilities, costs, rate limits, and task requirements.

Classes:
    ProviderRouter: Routes requests to optimal providers with fallback.
    NoProviderError: Raised when no provider is available for a capability.

Example:
    >>> from claudetube.providers.router import ProviderRouter
    >>> router = ProviderRouter()
    >>> transcriber = router.get_transcriber()
    >>> vision = router.get_vision_analyzer()
    >>> result = await router.call_with_fallback(
    ...     Capability.VISION, "analyze_images", images, prompt
    ... )
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from claudetube.providers.capabilities import PROVIDER_INFO, Capability, CostTier

if TYPE_CHECKING:
    from claudetube.providers.base import (
        Embedder,
        Provider,
        Reasoner,
        Transcriber,
        VideoAnalyzer,
        VisionAnalyzer,
    )
    from claudetube.providers.config import ProvidersConfig

logger = logging.getLogger(__name__)

# Capabilities where claude-code can serve as ultimate fallback
_CLAUDE_CODE_CAPABILITIES = frozenset({Capability.VISION, Capability.REASON})

# Maximum retries on 429 rate limit before falling back
_RATE_LIMIT_RETRIES = 2

# Base delay in seconds for rate limit backoff
_RATE_LIMIT_BASE_DELAY = 1.0


class NoProviderError(Exception):
    """Raised when no provider is available for a requested capability.

    This error indicates that neither the preferred provider, fallback chain,
    nor the claude-code ultimate fallback can satisfy the requested capability.

    Attributes:
        capability: The capability that could not be satisfied.

    Example:
        >>> try:
        ...     provider = router.get_for_capability(Capability.EMBED)
        ... except NoProviderError as e:
        ...     print(f"No provider for {e.capability}")
    """

    def __init__(self, capability: Capability, message: str | None = None):
        self.capability = capability
        if message is None:
            message = (
                f"No available provider for capability {capability.name}. "
                f"Check provider configuration and API keys."
            )
        super().__init__(message)


class ProviderRouter:
    """Routes requests to optimal providers based on capabilities.

    The ProviderRouter selects the best available provider for a given
    capability, following this order:

    1. Preferred provider from config (e.g., config.vision_provider)
    2. Fallback chain from config (e.g., config.vision_fallbacks)
    3. Claude Code as ultimate fallback (for VISION and REASON only)

    If no provider can be found, raises NoProviderError.

    Args:
        config: Provider configuration. If None, loads from
            get_providers_config().

    Example:
        >>> router = ProviderRouter()
        >>> transcriber = router.get_transcriber()
        >>> result = await transcriber.transcribe(audio_path)

        >>> # With custom config
        >>> config = ProvidersConfig(vision_provider="anthropic")
        >>> router = ProviderRouter(config=config)
    """

    def __init__(self, config: ProvidersConfig | None = None):
        if config is None:
            from claudetube.providers.config import get_providers_config

            config = get_providers_config()
        self._config = config

    def _get_preferred_provider_name(self, capability: Capability) -> str | None:
        """Get the preferred provider name for a capability from config.

        Args:
            capability: The capability to look up.

        Returns:
            Provider name string, or None if not configured.
        """
        mapping: dict[Capability, str | None] = {
            Capability.TRANSCRIBE: self._config.transcription_provider,
            Capability.VISION: self._config.vision_provider,
            Capability.VIDEO: self._config.video_provider,
            Capability.REASON: self._config.reasoning_provider,
            Capability.EMBED: self._config.embedding_provider,
        }
        return mapping.get(capability)

    def _get_fallback_chain(self, capability: Capability) -> list[str]:
        """Get the fallback chain for a capability from config.

        Args:
            capability: The capability to look up.

        Returns:
            List of provider names to try as fallbacks.
        """
        mapping: dict[Capability, list[str]] = {
            Capability.TRANSCRIBE: self._config.transcription_fallbacks,
            Capability.VISION: self._config.vision_fallbacks,
            Capability.REASON: self._config.reasoning_fallbacks,
        }
        return mapping.get(capability, [])

    def _try_load_provider(self, name: str) -> Provider | None:
        """Try to load a provider by name, returning None on failure.

        This method catches ImportError and ValueError so that
        unavailable providers are silently skipped during fallback.

        Args:
            name: Canonical provider name.

        Returns:
            Provider instance if loaded and available, None otherwise.
        """
        from claudetube.providers.registry import get_provider

        try:
            provider = get_provider(name)
        except (ImportError, ValueError) as e:
            logger.warning("Failed to load provider '%s': %s", name, e)
            return None

        if not provider.is_available():
            logger.warning(
                "Provider '%s' is not available (missing API key or dependencies)",
                name,
            )
            return None

        return provider

    def _sort_by_cost(self, names: list[str]) -> list[str]:
        """Sort provider names by cost tier (cheapest first).

        Providers not found in PROVIDER_INFO are assigned MODERATE tier.
        Uses a stable sort so config order is preserved within the same tier.

        Args:
            names: List of provider names.

        Returns:
            New list sorted by cost tier.
        """

        def _cost_key(name: str) -> int:
            info = PROVIDER_INFO.get(name)
            if info is not None:
                return info.cost_tier.value
            return CostTier.MODERATE.value

        return sorted(names, key=_cost_key)

    def get_for_capability(self, capability: Capability) -> Provider:
        """Get best available provider for a capability.

        Tries providers in this order:
        1. Preferred provider from config
        2. Fallback chain from config (sorted by cost if cost_preference="cost")
        3. Claude Code as ultimate fallback (VISION/REASON only)

        When cost_preference is "cost" (the default), fallback providers are
        sorted so cheaper providers are tried first. When "quality", the
        original config order is preserved.

        Args:
            capability: The capability needed.

        Returns:
            A Provider instance that supports the requested capability.

        Raises:
            NoProviderError: If no provider is available for the capability.

        Example:
            >>> provider = router.get_for_capability(Capability.VISION)
            >>> # provider is guaranteed to be available
        """
        # Track which providers we've already tried to avoid duplicates
        tried: set[str] = set()

        # 1. Try preferred provider
        preferred_name = self._get_preferred_provider_name(capability)
        if preferred_name is not None:
            tried.add(preferred_name)
            provider = self._try_load_provider(preferred_name)
            if provider is not None:
                logger.info(
                    "Selected preferred provider '%s' for %s",
                    preferred_name,
                    capability.name,
                )
                return provider
            logger.warning(
                "Preferred provider '%s' for %s not available, trying fallbacks",
                preferred_name,
                capability.name,
            )

        # 2. Try fallback chain (optionally sorted by cost)
        fallback_chain = self._get_fallback_chain(capability)
        if self._config.cost_preference == "cost":
            fallback_chain = self._sort_by_cost(fallback_chain)

        for fallback_name in fallback_chain:
            if fallback_name in tried:
                continue
            tried.add(fallback_name)
            provider = self._try_load_provider(fallback_name)
            if provider is not None:
                logger.info(
                    "Selected fallback provider '%s' for %s",
                    fallback_name,
                    capability.name,
                )
                return provider
            logger.warning(
                "Fallback provider '%s' for %s not available",
                fallback_name,
                capability.name,
            )

        # 3. Claude Code as ultimate fallback (VISION and REASON only)
        if capability in _CLAUDE_CODE_CAPABILITIES and "claude-code" not in tried:
            tried.add("claude-code")
            provider = self._try_load_provider("claude-code")
            if provider is not None:
                logger.info(
                    "Selected claude-code as ultimate fallback for %s",
                    capability.name,
                )
                return provider

        raise NoProviderError(capability)

    def get_transcriber(self) -> Transcriber:
        """Get a provider implementing the Transcriber protocol.

        Convenience method that routes to the best transcription provider
        and validates it implements the Transcriber protocol.

        Returns:
            A provider implementing the Transcriber protocol.

        Raises:
            NoProviderError: If no transcription provider is available.

        Example:
            >>> transcriber = router.get_transcriber()
            >>> result = await transcriber.transcribe(audio_path)
        """
        from claudetube.providers.base import Transcriber

        provider = self.get_for_capability(Capability.TRANSCRIBE)
        if not isinstance(provider, Transcriber):
            raise NoProviderError(
                Capability.TRANSCRIBE,
                f"Provider '{provider.info.name}' does not implement "
                f"Transcriber protocol",
            )
        return provider

    def get_vision_analyzer(self) -> VisionAnalyzer:
        """Get a provider implementing the VisionAnalyzer protocol.

        Convenience method that routes to the best vision provider
        and validates it implements the VisionAnalyzer protocol.

        Returns:
            A provider implementing the VisionAnalyzer protocol.

        Raises:
            NoProviderError: If no vision provider is available.

        Example:
            >>> analyzer = router.get_vision_analyzer()
            >>> result = await analyzer.analyze_images(images, prompt)
        """
        from claudetube.providers.base import VisionAnalyzer

        provider = self.get_for_capability(Capability.VISION)
        if not isinstance(provider, VisionAnalyzer):
            raise NoProviderError(
                Capability.VISION,
                f"Provider '{provider.info.name}' does not implement "
                f"VisionAnalyzer protocol",
            )
        return provider

    def get_video_analyzer(self) -> VideoAnalyzer | None:
        """Get a VideoAnalyzer provider if available.

        Video analysis is only supported by Gemini (Google). This method
        returns None if no video provider is configured or available,
        rather than raising an error.

        Returns:
            A provider implementing VideoAnalyzer, or None if unavailable.

        Example:
            >>> analyzer = router.get_video_analyzer()
            >>> if analyzer is not None:
            ...     result = await analyzer.analyze_video(video, prompt)
        """
        from claudetube.providers.base import VideoAnalyzer

        video_name = self._config.video_provider
        if video_name is None:
            logger.info(
                "No video provider configured (only Gemini supports native video)"
            )
            return None

        provider = self._try_load_provider(video_name)
        if provider is None:
            logger.warning(
                "Configured video provider '%s' is not available",
                video_name,
            )
            return None

        if not isinstance(provider, VideoAnalyzer):
            logger.warning(
                "Provider '%s' does not implement VideoAnalyzer protocol",
                video_name,
            )
            return None

        logger.info("Selected video provider '%s'", video_name)
        return provider

    def get_reasoner(self) -> Reasoner:
        """Get a provider implementing the Reasoner protocol.

        Convenience method that routes to the best reasoning provider
        and validates it implements the Reasoner protocol.

        Returns:
            A provider implementing the Reasoner protocol.

        Raises:
            NoProviderError: If no reasoning provider is available.

        Example:
            >>> reasoner = router.get_reasoner()
            >>> result = await reasoner.reason(messages)
        """
        from claudetube.providers.base import Reasoner

        provider = self.get_for_capability(Capability.REASON)
        if not isinstance(provider, Reasoner):
            raise NoProviderError(
                Capability.REASON,
                f"Provider '{provider.info.name}' does not implement Reasoner protocol",
            )
        return provider

    def get_embedder(self) -> Embedder:
        """Get a provider implementing the Embedder protocol.

        Convenience method that routes to the best embedding provider
        and validates it implements the Embedder protocol.

        Returns:
            A provider implementing the Embedder protocol.

        Raises:
            NoProviderError: If no embedding provider is available.

        Example:
            >>> embedder = router.get_embedder()
            >>> vector = await embedder.embed("some text")
        """
        from claudetube.providers.base import Embedder

        provider = self.get_for_capability(Capability.EMBED)
        if not isinstance(provider, Embedder):
            raise NoProviderError(
                Capability.EMBED,
                f"Provider '{provider.info.name}' does not implement Embedder protocol",
            )
        return provider

    def _is_parallel_fallback(self, capability: Capability) -> bool:
        """Check if parallel fallback is enabled for a capability.

        Args:
            capability: The capability to check.

        Returns:
            True if parallel fallback is configured for this capability.
        """
        # Map capability to config key names
        cap_names: dict[Capability, list[str]] = {
            Capability.TRANSCRIBE: ["transcription"],
            Capability.VISION: ["vision"],
            Capability.VIDEO: ["video"],
            Capability.REASON: ["reasoning"],
            Capability.EMBED: ["embedding"],
        }
        for name in cap_names.get(capability, []):
            if self._config.parallel_fallback.get(name, False):
                return True
        return False

    async def call_with_fallback(
        self,
        capability: Capability,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a provider method with automatic fallback on errors.

        Attempts to call the specified method on the preferred provider.
        On failure (rate limits, API errors, import errors), retries with
        exponential backoff for 429 errors, then falls back to the next
        provider in the chain.

        When ``parallel_fallback`` is enabled for the capability, all
        providers in the chain are called concurrently and the first
        successful result is returned. Remaining tasks are cancelled.

        Args:
            capability: The capability needed for this call.
            method: Name of the method to call on the provider.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The result of the provider method call.

        Raises:
            NoProviderError: If all providers fail or none are available.

        Example:
            >>> result = await router.call_with_fallback(
            ...     Capability.VISION,
            ...     "analyze_images",
            ...     [Path("frame.jpg")],
            ...     prompt="Describe this",
            ... )
        """
        # Build ordered list of providers to try
        providers_to_try = self._build_provider_list(capability)

        if not providers_to_try:
            raise NoProviderError(capability)

        # Parallel mode: call all providers concurrently, return first success
        if self._is_parallel_fallback(capability):
            return await self._call_parallel(
                capability, method, providers_to_try, *args, **kwargs
            )

        # Sequential mode (default)
        last_error: Exception | None = None

        for provider_name, provider in providers_to_try:
            fn = getattr(provider, method, None)
            if fn is None:
                logger.warning(
                    "Provider '%s' does not have method '%s'",
                    provider_name,
                    method,
                )
                continue

            # Try calling with rate limit retries
            for attempt in range(_RATE_LIMIT_RETRIES + 1):
                try:
                    logger.info(
                        "Calling %s.%s (attempt %d)",
                        provider_name,
                        method,
                        attempt + 1,
                    )
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as e:
                    last_error = e
                    if _is_rate_limit_error(e) and attempt < _RATE_LIMIT_RETRIES:
                        delay = _RATE_LIMIT_BASE_DELAY * (2**attempt)
                        logger.warning(
                            "Rate limited on '%s', retrying in %.1fs (attempt %d/%d)",
                            provider_name,
                            delay,
                            attempt + 1,
                            _RATE_LIMIT_RETRIES + 1,
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.warning(
                        "Provider '%s' failed for %s.%s: %s",
                        provider_name,
                        capability.name,
                        method,
                        e,
                    )
                    break  # Move to next provider

        raise NoProviderError(
            capability,
            f"All providers failed for {capability.name}.{method}. "
            f"Last error: {last_error}",
        )

    async def _call_parallel(
        self,
        capability: Capability,
        method: str,
        providers: list[tuple[str, Provider]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call multiple providers concurrently, return first successful result.

        Launches all provider calls as concurrent tasks. When the first one
        succeeds, its result is returned and all remaining tasks are cancelled.

        Args:
            capability: The capability being invoked.
            method: Method name to call on each provider.
            providers: List of (name, provider) tuples.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Result from the first successful provider.

        Raises:
            NoProviderError: If all providers fail.
        """
        # Filter to providers that have the method
        callable_providers = []
        for name, provider in providers:
            fn = getattr(provider, method, None)
            if fn is not None:
                callable_providers.append((name, fn))
            else:
                logger.warning("Provider '%s' does not have method '%s'", name, method)

        if not callable_providers:
            raise NoProviderError(capability)

        logger.info(
            "Parallel fallback: calling %d providers for %s.%s",
            len(callable_providers),
            capability.name,
            method,
        )

        async def _try_provider(name: str, fn: Any) -> tuple[str, Any]:
            """Call a single provider and return (name, result)."""
            logger.info("Parallel: calling %s.%s", name, method)
            result = await fn(*args, **kwargs)
            return name, result

        # Create tasks for all providers
        tasks = [
            asyncio.create_task(_try_provider(name, fn), name=f"parallel-{name}")
            for name, fn in callable_providers
        ]

        # Wait for first success, collect errors
        errors: list[tuple[str, Exception]] = []
        try:
            while tasks:
                done, tasks_set = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                tasks = list(tasks_set)

                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        # Extract provider name from task name
                        provider_name = task.get_name().removeprefix("parallel-")
                        logger.warning(
                            "Parallel: provider '%s' failed: %s",
                            provider_name,
                            exc,
                        )
                        errors.append((provider_name, exc))
                        continue

                    # Success! Cancel remaining tasks and return
                    winner_name, result = task.result()
                    logger.info(
                        "Parallel: '%s' succeeded first for %s.%s",
                        winner_name,
                        capability.name,
                        method,
                    )
                    for remaining in tasks:
                        remaining.cancel()
                    return result
        finally:
            # Ensure all tasks are cleaned up
            for task in tasks:
                task.cancel()
            # Suppress CancelledError from cancelled tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # All providers failed
        last_name, last_error = errors[-1] if errors else ("unknown", None)
        raise NoProviderError(
            capability,
            f"All {len(callable_providers)} parallel providers failed for "
            f"{capability.name}.{method}. Last error from '{last_name}': "
            f"{last_error}",
        )

    def _build_provider_list(
        self, capability: Capability
    ) -> list[tuple[str, Provider]]:
        """Build an ordered list of (name, provider) tuples to try.

        Deduplicates and filters to only available providers. When
        cost_preference is "cost", the preferred provider stays first
        but fallback providers are sorted by cost tier.

        Args:
            capability: The capability needed.

        Returns:
            List of (provider_name, provider_instance) tuples.
        """
        tried: set[str] = set()
        result: list[tuple[str, Provider]] = []

        # 1. Preferred provider always first
        preferred = self._get_preferred_provider_name(capability)
        if preferred is not None:
            tried.add(preferred)
            provider = self._try_load_provider(preferred)
            if provider is not None:
                result.append((preferred, provider))

        # 2. Fallback chain (optionally sorted by cost)
        fallback_chain = self._get_fallback_chain(capability)
        if self._config.cost_preference == "cost":
            fallback_chain = self._sort_by_cost(fallback_chain)

        for name in fallback_chain:
            if name in tried:
                continue
            tried.add(name)
            provider = self._try_load_provider(name)
            if provider is not None:
                result.append((name, provider))

        # 3. Claude-code ultimate fallback for VISION/REASON
        if capability in _CLAUDE_CODE_CAPABILITIES and "claude-code" not in tried:
            tried.add("claude-code")
            provider = self._try_load_provider("claude-code")
            if provider is not None:
                result.append(("claude-code", provider))

        return result


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception represents a 429 rate limit error.

    Checks for common HTTP status code patterns across different
    provider SDKs.

    Args:
        error: The exception to check.

    Returns:
        True if the error represents a rate limit (429) response.
    """
    # Check common status_code attribute
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    # Check for httpx-style response attribute
    response = getattr(error, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status == 429:
            return True

    # Check error message for 429
    error_str = str(error).lower()
    return "429" in error_str or "rate limit" in error_str


__all__ = [
    "NoProviderError",
    "ProviderRouter",
]
