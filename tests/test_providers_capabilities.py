"""Tests for provider capabilities and ProviderInfo.

Verifies that:
1. Capability enum contains all expected values
2. ProviderInfo is immutable (frozen dataclass)
3. can(), can_all(), can_any() methods work correctly
4. Pre-defined PROVIDER_INFO contains all expected providers
"""

from __future__ import annotations

import pytest

from claudetube.providers.capabilities import (
    PROVIDER_INFO,
    Capability,
    ProviderInfo,
)


class TestCapabilityEnum:
    """Tests for the Capability enum."""

    def test_all_capabilities_defined(self):
        """All expected capabilities are defined."""
        assert Capability.TRANSCRIBE is not None
        assert Capability.VISION is not None
        assert Capability.VIDEO is not None
        assert Capability.REASON is not None
        assert Capability.EMBED is not None

    def test_capability_count(self):
        """Exactly 5 capabilities are defined."""
        assert len(Capability) == 5

    def test_capabilities_are_unique(self):
        """Each capability has a unique value."""
        values = [c.value for c in Capability]
        assert len(values) == len(set(values))

    def test_capabilities_can_be_in_frozenset(self):
        """Capabilities can be added to a frozenset."""
        caps = frozenset({Capability.TRANSCRIBE, Capability.VISION})
        assert Capability.TRANSCRIBE in caps
        assert Capability.VISION in caps
        assert Capability.VIDEO not in caps


class TestProviderInfo:
    """Tests for the ProviderInfo dataclass."""

    def test_basic_instantiation(self):
        """ProviderInfo can be instantiated with required fields."""
        info = ProviderInfo(
            name="test-provider",
            capabilities=frozenset({Capability.REASON}),
        )
        assert info.name == "test-provider"
        assert Capability.REASON in info.capabilities

    def test_default_values(self):
        """ProviderInfo has sensible defaults."""
        info = ProviderInfo(
            name="test-provider",
            capabilities=frozenset(),
        )
        assert info.supports_structured_output is False
        assert info.supports_streaming is False
        assert info.max_audio_size_mb is None
        assert info.max_images_per_request is None
        assert info.cost_per_1m_input_tokens is None

    def test_is_immutable(self):
        """ProviderInfo is frozen and cannot be modified."""
        info = ProviderInfo(
            name="test-provider",
            capabilities=frozenset({Capability.REASON}),
        )
        with pytest.raises(AttributeError):
            info.name = "modified"  # type: ignore

    def test_full_instantiation(self):
        """ProviderInfo can be instantiated with all fields."""
        info = ProviderInfo(
            name="full-provider",
            capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION}),
            supports_structured_output=True,
            supports_streaming=True,
            max_audio_size_mb=25.0,
            max_audio_duration_sec=1500.0,
            supports_diarization=True,
            supports_translation=True,
            max_images_per_request=10,
            max_image_size_mb=5.0,
            max_video_duration_sec=7200.0,
            max_video_size_mb=2000.0,
            video_tokens_per_second=300.0,
            max_context_tokens=128_000,
            cost_per_1m_input_tokens=2.50,
            cost_per_1m_output_tokens=10.00,
            cost_per_minute_audio=0.006,
        )
        assert info.name == "full-provider"
        assert info.supports_structured_output is True
        assert info.max_audio_size_mb == 25.0
        assert info.cost_per_1m_input_tokens == 2.50


class TestProviderInfoCanMethod:
    """Tests for ProviderInfo.can() method."""

    @pytest.fixture
    def multi_cap_info(self):
        """Create a ProviderInfo with multiple capabilities."""
        return ProviderInfo(
            name="multi-cap",
            capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION, Capability.REASON}),
        )

    def test_can_with_present_capability(self, multi_cap_info):
        """can() returns True for present capabilities."""
        assert multi_cap_info.can(Capability.TRANSCRIBE) is True
        assert multi_cap_info.can(Capability.VISION) is True
        assert multi_cap_info.can(Capability.REASON) is True

    def test_can_with_absent_capability(self, multi_cap_info):
        """can() returns False for absent capabilities."""
        assert multi_cap_info.can(Capability.VIDEO) is False
        assert multi_cap_info.can(Capability.EMBED) is False

    def test_can_with_empty_capabilities(self):
        """can() returns False for all capabilities when empty."""
        info = ProviderInfo(name="empty", capabilities=frozenset())
        for cap in Capability:
            assert info.can(cap) is False


class TestProviderInfoCanAllMethod:
    """Tests for ProviderInfo.can_all() method."""

    @pytest.fixture
    def multi_cap_info(self):
        """Create a ProviderInfo with multiple capabilities."""
        return ProviderInfo(
            name="multi-cap",
            capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION, Capability.REASON}),
        )

    def test_can_all_with_all_present(self, multi_cap_info):
        """can_all() returns True when all capabilities present."""
        assert multi_cap_info.can_all(Capability.TRANSCRIBE, Capability.VISION) is True
        assert multi_cap_info.can_all(Capability.TRANSCRIBE, Capability.VISION, Capability.REASON) is True

    def test_can_all_with_some_absent(self, multi_cap_info):
        """can_all() returns False when any capability is absent."""
        assert multi_cap_info.can_all(Capability.TRANSCRIBE, Capability.VIDEO) is False
        assert multi_cap_info.can_all(Capability.VISION, Capability.EMBED) is False

    def test_can_all_with_single_capability(self, multi_cap_info):
        """can_all() works with single capability (same as can())."""
        assert multi_cap_info.can_all(Capability.TRANSCRIBE) is True
        assert multi_cap_info.can_all(Capability.VIDEO) is False

    def test_can_all_with_no_capabilities(self, multi_cap_info):
        """can_all() returns True with no arguments (vacuous truth)."""
        assert multi_cap_info.can_all() is True


class TestProviderInfoCanAnyMethod:
    """Tests for ProviderInfo.can_any() method."""

    @pytest.fixture
    def multi_cap_info(self):
        """Create a ProviderInfo with multiple capabilities."""
        return ProviderInfo(
            name="multi-cap",
            capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION}),
        )

    def test_can_any_with_some_present(self, multi_cap_info):
        """can_any() returns True when any capability present."""
        assert multi_cap_info.can_any(Capability.TRANSCRIBE, Capability.VIDEO) is True
        assert multi_cap_info.can_any(Capability.EMBED, Capability.VISION) is True

    def test_can_any_with_none_present(self, multi_cap_info):
        """can_any() returns False when no capabilities present."""
        assert multi_cap_info.can_any(Capability.VIDEO, Capability.EMBED) is False
        assert multi_cap_info.can_any(Capability.REASON) is False

    def test_can_any_with_all_present(self, multi_cap_info):
        """can_any() returns True when all capabilities present."""
        assert multi_cap_info.can_any(Capability.TRANSCRIBE, Capability.VISION) is True

    def test_can_any_with_no_capabilities(self, multi_cap_info):
        """can_any() returns False with no arguments."""
        assert multi_cap_info.can_any() is False


class TestProviderInfoHashability:
    """Tests for ProviderInfo hashability (required for use in sets/dicts)."""

    def test_is_hashable(self):
        """ProviderInfo can be hashed."""
        info = ProviderInfo(
            name="test-provider",
            capabilities=frozenset({Capability.REASON}),
        )
        # Should not raise
        hash(info)

    def test_can_be_in_set(self):
        """ProviderInfo instances can be added to a set."""
        info1 = ProviderInfo(name="p1", capabilities=frozenset({Capability.REASON}))
        info2 = ProviderInfo(name="p2", capabilities=frozenset({Capability.VISION}))
        info_set = {info1, info2}
        assert len(info_set) == 2

    def test_equal_instances_have_same_hash(self):
        """Equal ProviderInfo instances have the same hash."""
        info1 = ProviderInfo(name="test", capabilities=frozenset({Capability.REASON}))
        info2 = ProviderInfo(name="test", capabilities=frozenset({Capability.REASON}))
        assert info1 == info2
        assert hash(info1) == hash(info2)


class TestPreDefinedProviderInfo:
    """Tests for the pre-defined PROVIDER_INFO dictionary."""

    def test_expected_providers_defined(self):
        """All expected providers are in PROVIDER_INFO."""
        expected = [
            "whisper-local",
            "openai",
            "anthropic",
            "google",
            "deepgram",
            "assemblyai",
            "claude-code",
            "ollama",
            "voyage",
        ]
        for name in expected:
            assert name in PROVIDER_INFO, f"Expected provider '{name}' not found"

    def test_whisper_local_capabilities(self):
        """whisper-local has correct capabilities."""
        info = PROVIDER_INFO["whisper-local"]
        assert info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.VISION)
        assert not info.can(Capability.VIDEO)
        assert not info.can(Capability.REASON)
        assert not info.can(Capability.EMBED)
        assert info.cost_per_minute_audio == 0  # Free

    def test_openai_capabilities(self):
        """openai has correct capabilities."""
        info = PROVIDER_INFO["openai"]
        assert info.can(Capability.TRANSCRIBE)
        assert info.can(Capability.VISION)
        assert info.can(Capability.REASON)
        assert not info.can(Capability.VIDEO)
        assert not info.can(Capability.EMBED)
        assert info.supports_structured_output is True
        assert info.max_audio_size_mb == 25

    def test_anthropic_capabilities(self):
        """anthropic has correct capabilities."""
        info = PROVIDER_INFO["anthropic"]
        assert info.can(Capability.VISION)
        assert info.can(Capability.REASON)
        assert not info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.VIDEO)
        assert not info.can(Capability.EMBED)
        assert info.supports_structured_output is True

    def test_google_capabilities(self):
        """google has correct capabilities including VIDEO."""
        info = PROVIDER_INFO["google"]
        assert info.can(Capability.VISION)
        assert info.can(Capability.VIDEO)  # Unique to Google
        assert info.can(Capability.REASON)
        assert not info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.EMBED)
        assert info.max_video_duration_sec == 7200  # 2 hours

    def test_deepgram_capabilities(self):
        """deepgram has correct capabilities."""
        info = PROVIDER_INFO["deepgram"]
        assert info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.VISION)
        assert info.supports_diarization is True

    def test_claude_code_capabilities(self):
        """claude-code has correct capabilities."""
        info = PROVIDER_INFO["claude-code"]
        assert info.can(Capability.VISION)
        assert info.can(Capability.REASON)
        assert not info.can(Capability.TRANSCRIBE)
        assert info.cost_per_1m_input_tokens == 0  # Included with subscription

    def test_ollama_capabilities(self):
        """ollama has correct capabilities."""
        info = PROVIDER_INFO["ollama"]
        assert info.can(Capability.VISION)
        assert info.can(Capability.REASON)
        assert info.max_images_per_request == 1  # LLaVA limitation
        assert info.cost_per_1m_input_tokens == 0  # Local

    def test_voyage_capabilities(self):
        """voyage has correct capabilities."""
        info = PROVIDER_INFO["voyage"]
        assert info.can(Capability.EMBED)
        assert not info.can(Capability.VISION)
        assert not info.can(Capability.REASON)

    def test_all_providers_have_names_matching_keys(self):
        """All provider info names match their dictionary keys."""
        for name, info in PROVIDER_INFO.items():
            assert info.name == name, f"Mismatch: key={name}, info.name={info.name}"


class TestCapabilityExports:
    """Tests for module exports."""

    def test_all_exported(self):
        """All expected items are in __all__."""
        from claudetube.providers import capabilities

        assert "Capability" in capabilities.__all__
        assert "ProviderInfo" in capabilities.__all__
        assert "PROVIDER_INFO" in capabilities.__all__

    def test_can_import_from_module(self):
        """Can import directly from capabilities module."""
        from claudetube.providers.capabilities import (
            PROVIDER_INFO,
            Capability,
            ProviderInfo,
        )

        assert Capability is not None
        assert ProviderInfo is not None
        assert PROVIDER_INFO is not None
