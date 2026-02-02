"""Tests for ClaudeCodeProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior with env vars
3. analyze_images() formatting
4. reason() message formatting
5. Schema formatting for structured output
6. Registry integration
7. Protocol compliance (VisionAnalyzer, Reasoner)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.providers.base import Reasoner, VisionAnalyzer
from claudetube.providers.capabilities import Capability
from claudetube.providers.claude_code.client import ClaudeCodeProvider


class TestClaudeCodeProviderInfo:
    """Tests for provider metadata."""

    def test_info_name(self):
        provider = ClaudeCodeProvider()
        assert provider.info.name == "claude-code"

    def test_info_capabilities(self):
        provider = ClaudeCodeProvider()
        assert provider.info.can(Capability.VISION)
        assert provider.info.can(Capability.REASON)
        assert not provider.info.can(Capability.TRANSCRIBE)
        assert not provider.info.can(Capability.VIDEO)
        assert not provider.info.can(Capability.EMBED)

    def test_info_structured_output(self):
        provider = ClaudeCodeProvider()
        assert provider.info.supports_structured_output is True

    def test_info_no_streaming(self):
        provider = ClaudeCodeProvider()
        assert provider.info.supports_streaming is False

    def test_info_zero_cost(self):
        provider = ClaudeCodeProvider()
        assert provider.info.cost_per_1m_input_tokens == 0
        assert provider.info.cost_per_1m_output_tokens == 0


class TestIsAvailable:
    """Tests for is_available() detection logic."""

    def test_default_true(self):
        """Should return True by default (safe fallback)."""
        with patch.dict("os.environ", {}, clear=True):
            provider = ClaudeCodeProvider()
            assert provider.is_available() is True

    def test_mcp_server_env(self):
        with patch.dict("os.environ", {"MCP_SERVER": "1"}):
            provider = ClaudeCodeProvider()
            assert provider.is_available() is True

    def test_claude_code_env(self):
        with patch.dict("os.environ", {"CLAUDE_CODE": "1"}):
            provider = ClaudeCodeProvider()
            assert provider.is_available() is True

    def test_both_env_vars(self):
        with patch.dict("os.environ", {"MCP_SERVER": "1", "CLAUDE_CODE": "1"}):
            provider = ClaudeCodeProvider()
            assert provider.is_available() is True

    def test_no_env_vars_still_available(self):
        """Even without env vars, defaults to True."""
        with patch.dict("os.environ", {}, clear=True):
            provider = ClaudeCodeProvider()
            assert provider.is_available() is True


class TestAnalyzeImages:
    """Tests for analyze_images() formatting."""

    @pytest.mark.asyncio
    async def test_single_image(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake image")

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([img], "What is shown?")

        assert f"[Image: {img.resolve()}]" in result
        assert "What is shown?" in result

    @pytest.mark.asyncio
    async def test_multiple_images(self, tmp_path):
        imgs = []
        for i in range(3):
            img = tmp_path / f"frame{i}.jpg"
            img.write_bytes(b"fake image")
            imgs.append(img)

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images(imgs, "Describe these frames")

        for img in imgs:
            assert f"[Image: {img.resolve()}]" in result
        assert "Describe these frames" in result

    @pytest.mark.asyncio
    async def test_missing_image(self, tmp_path):
        img = tmp_path / "nonexistent.jpg"

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([img], "What is this?")

        assert "[Image not found:" in result

    @pytest.mark.asyncio
    async def test_mixed_existing_and_missing(self, tmp_path):
        existing = tmp_path / "exists.jpg"
        existing.write_bytes(b"fake image")
        missing = tmp_path / "missing.jpg"

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([existing, missing], "Describe")

        assert f"[Image: {existing.resolve()}]" in result
        assert "[Image not found:" in result

    @pytest.mark.asyncio
    async def test_empty_images(self):
        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([], "Describe nothing")

        assert "Describe nothing" in result

    @pytest.mark.asyncio
    async def test_with_pydantic_schema(self, tmp_path):
        """Schema is formatted as JSON in the output."""
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake image")

        try:
            from pydantic import BaseModel, Field

            class TestSchema(BaseModel):
                description: str = Field(description="Scene description")
                objects: list[str] = Field(default_factory=list)

            provider = ClaudeCodeProvider()
            result = await provider.analyze_images(
                [img], "Extract objects", schema=TestSchema
            )

            assert "Respond with JSON matching this schema:" in result
            assert "```json" in result
            assert "description" in result
            assert "objects" in result
        except ImportError:
            pytest.skip("pydantic not installed")

    @pytest.mark.asyncio
    async def test_without_schema(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake image")

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([img], "Describe this")

        assert "Respond with JSON matching this schema:" not in result
        assert "```json" not in result

    @pytest.mark.asyncio
    async def test_returns_string(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake image")

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([img], "What is this?")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_absolute_paths(self, tmp_path):
        """Image paths in output should be absolute."""
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake image")

        provider = ClaudeCodeProvider()
        result = await provider.analyze_images([img], "Describe")

        # Extract path from [Image: /path/to/file]
        for line in result.split("\n"):
            if line.startswith("[Image: "):
                path_str = line[len("[Image: ") : -1]
                assert Path(path_str).is_absolute()


class TestReason:
    """Tests for reason() message formatting."""

    @pytest.mark.asyncio
    async def test_user_message(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "user", "content": "Hello"},
            ]
        )

        assert "Hello" in result
        assert "[System]" not in result

    @pytest.mark.asyncio
    async def test_system_message(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "system", "content": "You are helpful"},
            ]
        )

        assert "[System]: You are helpful" in result

    @pytest.mark.asyncio
    async def test_assistant_message(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "assistant", "content": "I can help"},
            ]
        )

        assert "[Previous response]: I can help" in result

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Summarize this video"},
                {"role": "assistant", "content": "The video shows..."},
                {"role": "user", "content": "What about the ending?"},
            ]
        )

        assert "[System]: Be concise" in result
        assert "Summarize this video" in result
        assert "[Previous response]: The video shows..." in result
        assert "What about the ending?" in result

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason([])

        assert isinstance(result, str)
        assert result == ""

    @pytest.mark.asyncio
    async def test_default_role_is_user(self):
        """Messages without role are treated as user messages."""
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"content": "No role specified"},
            ]
        )

        assert "No role specified" in result
        assert "[System]" not in result
        assert "[Previous response]" not in result

    @pytest.mark.asyncio
    async def test_with_pydantic_schema(self):
        try:
            from pydantic import BaseModel, Field

            class Summary(BaseModel):
                title: str = Field(description="Video title")
                key_points: list[str] = Field(default_factory=list)

            provider = ClaudeCodeProvider()
            result = await provider.reason(
                [{"role": "user", "content": "Summarize"}],
                schema=Summary,
            )

            assert "Respond with JSON matching this schema:" in result
            assert "```json" in result
            assert "title" in result
            assert "key_points" in result
        except ImportError:
            pytest.skip("pydantic not installed")

    @pytest.mark.asyncio
    async def test_without_schema(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "user", "content": "Hello"},
            ]
        )

        assert "schema" not in result.lower()
        assert "```json" not in result

    @pytest.mark.asyncio
    async def test_returns_string(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "user", "content": "Hello"},
            ]
        )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_messages_separated_by_double_newline(self):
        provider = ClaudeCodeProvider()
        result = await provider.reason(
            [
                {"role": "system", "content": "System msg"},
                {"role": "user", "content": "User msg"},
            ]
        )

        assert "\n\n" in result


class TestSchemaFormatting:
    """Tests for _get_schema_json static method."""

    def test_pydantic_model(self):
        try:
            from pydantic import BaseModel, Field

            class TestModel(BaseModel):
                name: str = Field(description="The name")
                count: int = Field(default=0)

            result = ClaudeCodeProvider._get_schema_json(TestModel)
            parsed = json.loads(result)
            assert "properties" in parsed
            assert "name" in parsed["properties"]
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_non_pydantic_type(self):
        """Falls back to str() for non-Pydantic types."""
        result = ClaudeCodeProvider._get_schema_json(dict)
        assert "dict" in result.lower()


class TestProtocolCompliance:
    """Tests that ClaudeCodeProvider satisfies protocol contracts."""

    def test_implements_vision_analyzer(self):
        provider = ClaudeCodeProvider()
        assert isinstance(provider, VisionAnalyzer)

    def test_implements_reasoner(self):
        provider = ClaudeCodeProvider()
        assert isinstance(provider, Reasoner)


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("claude-code")
        assert isinstance(provider, ClaudeCodeProvider)
        assert provider.info.name == "claude-code"

    def test_get_provider_caching(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        p1 = get_provider("claude-code")
        p2 = get_provider("claude-code")
        assert p1 is p2

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("claude-code")
        assert isinstance(provider, ClaudeCodeProvider)
