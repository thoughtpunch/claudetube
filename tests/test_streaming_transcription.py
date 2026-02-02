"""Tests for streaming transcription protocol and DeepgramProvider streaming."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.base import StreamingTranscriber
from claudetube.providers.deepgram.client import DeepgramProvider
from claudetube.providers.types import (
    StreamingEventType,
    StreamingTranscriptionEvent,
    TranscriptionSegment,
)


class TestStreamingTranscriptionEvent:
    """Tests for StreamingTranscriptionEvent dataclass."""

    def test_partial_event(self):
        segment = TranscriptionSegment(start=0.0, end=1.5, text="hello")
        event = StreamingTranscriptionEvent(
            event_type=StreamingEventType.PARTIAL,
            segment=segment,
            is_final=False,
        )
        assert event.event_type == StreamingEventType.PARTIAL
        assert event.segment is not None
        assert event.segment.text == "hello"
        assert event.is_final is False
        assert event.accumulated_text == ""

    def test_final_event(self):
        segment = TranscriptionSegment(start=0.0, end=1.5, text="hello world")
        event = StreamingTranscriptionEvent(
            event_type=StreamingEventType.FINAL,
            segment=segment,
            is_final=True,
            accumulated_text="hello world",
        )
        assert event.event_type == StreamingEventType.FINAL
        assert event.is_final is True
        assert event.accumulated_text == "hello world"

    def test_complete_event(self):
        event = StreamingTranscriptionEvent(
            event_type=StreamingEventType.COMPLETE,
            accumulated_text="full transcript here",
        )
        assert event.event_type == StreamingEventType.COMPLETE
        assert event.segment is None
        assert event.accumulated_text == "full transcript here"

    def test_error_event(self):
        event = StreamingTranscriptionEvent(
            event_type=StreamingEventType.ERROR,
            error="Connection lost",
        )
        assert event.event_type == StreamingEventType.ERROR
        assert event.error == "Connection lost"
        assert event.segment is None


class TestStreamingEventType:
    """Tests for StreamingEventType enum."""

    def test_all_event_types_exist(self):
        assert StreamingEventType.PARTIAL
        assert StreamingEventType.FINAL
        assert StreamingEventType.COMPLETE
        assert StreamingEventType.ERROR

    def test_event_types_are_distinct(self):
        types = [
            StreamingEventType.PARTIAL,
            StreamingEventType.FINAL,
            StreamingEventType.COMPLETE,
            StreamingEventType.ERROR,
        ]
        assert len(set(types)) == 4


class TestStreamingTranscriberProtocol:
    """Tests for the StreamingTranscriber protocol."""

    def test_deepgram_implements_protocol(self):
        """DeepgramProvider should implement StreamingTranscriber."""
        provider = DeepgramProvider(api_key="test-key")
        assert isinstance(provider, StreamingTranscriber)

    def test_has_stream_transcribe_method(self):
        provider = DeepgramProvider(api_key="test-key")
        assert hasattr(provider, "stream_transcribe")
        assert callable(provider.stream_transcribe)


class TestDeepgramStreamTranscribe:
    """Tests for DeepgramProvider.stream_transcribe."""

    @pytest.mark.asyncio
    async def test_file_not_found_raises(self):
        provider = DeepgramProvider(api_key="test-key")
        with pytest.raises(FileNotFoundError):
            async for _ in provider.stream_transcribe(Path("/nonexistent/audio.mp3")):
                pass

    @pytest.mark.asyncio
    async def test_stream_yields_complete_event(self, tmp_path):
        """Streaming should yield at minimum a COMPLETE event."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 1024)

        provider = DeepgramProvider(api_key="test-key")

        mock_connection = AsyncMock()
        mock_connection.start = AsyncMock(return_value=True)
        mock_connection.send = AsyncMock()
        mock_connection.finish = AsyncMock()
        mock_connection.on = MagicMock()

        mock_client = MagicMock()
        mock_client.listen.asynclive.v.return_value = mock_connection

        # Mock the deepgram module imports used inside stream_transcribe
        mock_live_options = MagicMock()
        mock_live_events = MagicMock()
        mock_live_events.Transcript = "Transcript"
        mock_live_events.Error = "Error"

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.dict(
                "sys.modules",
                {
                    "deepgram": MagicMock(
                        LiveOptions=mock_live_options,
                        LiveTranscriptionEvents=mock_live_events,
                    ),
                },
            ),
        ):
            events: list[StreamingTranscriptionEvent] = []
            async for event in provider.stream_transcribe(audio_file):
                events.append(event)

        # Should at least have a COMPLETE event
        assert len(events) >= 1
        assert events[-1].event_type == StreamingEventType.COMPLETE

    @pytest.mark.asyncio
    async def test_stream_connection_failure(self, tmp_path):
        """Failed connection should yield an ERROR event."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 1024)

        provider = DeepgramProvider(api_key="test-key")

        mock_connection = AsyncMock()
        mock_connection.start = AsyncMock(return_value=False)
        mock_connection.on = MagicMock()

        mock_client = MagicMock()
        mock_client.listen.asynclive.v.return_value = mock_connection

        mock_live_options = MagicMock()
        mock_live_events = MagicMock()
        mock_live_events.Transcript = "Transcript"
        mock_live_events.Error = "Error"

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.dict(
                "sys.modules",
                {
                    "deepgram": MagicMock(
                        LiveOptions=mock_live_options,
                        LiveTranscriptionEvents=mock_live_events,
                    ),
                },
            ),
        ):
            events: list[StreamingTranscriptionEvent] = []
            async for event in provider.stream_transcribe(audio_file):
                events.append(event)

        assert len(events) == 1
        assert events[0].event_type == StreamingEventType.ERROR
        assert "Failed to start" in (events[0].error or "")

    @pytest.mark.asyncio
    async def test_stream_accumulates_text(self, tmp_path):
        """FINAL events should accumulate text in order."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        provider = DeepgramProvider(api_key="test-key")

        handlers: dict[str, list] = {}

        mock_connection = AsyncMock()
        mock_connection.start = AsyncMock(return_value=True)
        mock_connection.finish = AsyncMock()

        call_count = 0

        async def mock_send(data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate a transcript callback
                for key, handler_list in handlers.items():
                    if "Transcript" in str(key):
                        for handler in handler_list:
                            mock_result = MagicMock()
                            mock_result.channel.alternatives = [MagicMock()]
                            mock_result.channel.alternatives[
                                0
                            ].transcript = "hello world"
                            mock_result.channel.alternatives[0].confidence = 0.95
                            mock_result.is_final = True
                            mock_result.start = 0.0
                            mock_result.duration = 1.5
                            handler(None, mock_result)

        mock_connection.send = mock_send

        def mock_on(event_type, handler):
            handlers[str(event_type)] = handlers.get(str(event_type), [])
            handlers[str(event_type)].append(handler)

        mock_connection.on = mock_on

        mock_client = MagicMock()
        mock_client.listen.asynclive.v.return_value = mock_connection

        mock_live_options = MagicMock()
        mock_live_events = MagicMock()
        mock_live_events.Transcript = "Transcript"
        mock_live_events.Error = "Error"

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.dict(
                "sys.modules",
                {
                    "deepgram": MagicMock(
                        LiveOptions=mock_live_options,
                        LiveTranscriptionEvents=mock_live_events,
                    ),
                },
            ),
        ):
            events: list[StreamingTranscriptionEvent] = []
            async for event in provider.stream_transcribe(audio_file):
                events.append(event)

        final_events = [e for e in events if e.event_type == StreamingEventType.FINAL]
        complete_events = [
            e for e in events if e.event_type == StreamingEventType.COMPLETE
        ]

        assert len(final_events) >= 1
        assert final_events[0].segment is not None
        assert final_events[0].segment.text == "hello world"
        assert final_events[0].accumulated_text == "hello world"

        assert len(complete_events) == 1
        assert complete_events[0].accumulated_text == "hello world"


class TestStreamingTranscriberExports:
    """Tests for module exports."""

    def test_protocol_exported_from_base(self):
        from claudetube.providers.base import StreamingTranscriber

        assert StreamingTranscriber is not None

    def test_event_types_exported_from_types(self):
        from claudetube.providers.types import (
            StreamingEventType,
            StreamingTranscriptionEvent,
        )

        assert StreamingEventType is not None
        assert StreamingTranscriptionEvent is not None

    def test_streaming_event_type_in_all(self):
        from claudetube.providers import types

        assert "StreamingEventType" in types.__all__
        assert "StreamingTranscriptionEvent" in types.__all__

    def test_streaming_transcriber_in_all(self):
        from claudetube.providers import base

        assert "StreamingTranscriber" in base.__all__
