"""Tests for provider result types and schemas.

Verifies that:
1. TranscriptionSegment and TranscriptionResult work correctly
2. Format conversion methods (to_srt, to_vtt, to_dict) produce valid output
3. Round-trip serialization works (to_dict -> from_dict)
4. Pydantic models serialize to JSON Schema when available
"""

from __future__ import annotations

import pytest

from claudetube.providers.types import (
    TranscriptionResult,
    TranscriptionSegment,
)


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_basic_creation(self):
        """Can create a basic segment with required fields."""
        segment = TranscriptionSegment(
            start=0.0,
            end=1.5,
            text="Hello world",
        )
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.text == "Hello world"
        assert segment.confidence is None
        assert segment.speaker is None

    def test_creation_with_optional_fields(self):
        """Can create a segment with all fields."""
        segment = TranscriptionSegment(
            start=10.5,
            end=15.2,
            text="Welcome to the video",
            confidence=0.95,
            speaker="SPEAKER_00",
        )
        assert segment.start == 10.5
        assert segment.end == 15.2
        assert segment.text == "Welcome to the video"
        assert segment.confidence == 0.95
        assert segment.speaker == "SPEAKER_00"

    def test_to_dict(self):
        """to_dict() returns all fields as dictionary."""
        segment = TranscriptionSegment(
            start=1.0,
            end=2.0,
            text="Test",
            confidence=0.8,
            speaker="SPEAKER_01",
        )
        d = segment.to_dict()
        assert d == {
            "start": 1.0,
            "end": 2.0,
            "text": "Test",
            "confidence": 0.8,
            "speaker": "SPEAKER_01",
        }

    def test_to_dict_with_none_values(self):
        """to_dict() includes None values for optional fields."""
        segment = TranscriptionSegment(start=0.0, end=1.0, text="Hi")
        d = segment.to_dict()
        assert "confidence" in d
        assert d["confidence"] is None
        assert "speaker" in d
        assert d["speaker"] is None

    def test_from_dict(self):
        """from_dict() creates segment from dictionary."""
        data = {
            "start": 5.0,
            "end": 10.0,
            "text": "From dict",
            "confidence": 0.9,
            "speaker": "SPEAKER_02",
        }
        segment = TranscriptionSegment.from_dict(data)
        assert segment.start == 5.0
        assert segment.end == 10.0
        assert segment.text == "From dict"
        assert segment.confidence == 0.9
        assert segment.speaker == "SPEAKER_02"

    def test_from_dict_minimal(self):
        """from_dict() works with only required fields."""
        data = {"start": 0.0, "end": 1.0, "text": "Minimal"}
        segment = TranscriptionSegment.from_dict(data)
        assert segment.text == "Minimal"
        assert segment.confidence is None
        assert segment.speaker is None

    def test_round_trip(self):
        """Round-trip: segment -> to_dict -> from_dict -> matches original."""
        original = TranscriptionSegment(
            start=12.5,
            end=17.3,
            text="Round trip test",
            confidence=0.85,
            speaker="HOST",
        )
        reconstructed = TranscriptionSegment.from_dict(original.to_dict())
        assert reconstructed.start == original.start
        assert reconstructed.end == original.end
        assert reconstructed.text == original.text
        assert reconstructed.confidence == original.confidence
        assert reconstructed.speaker == original.speaker


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_basic_creation(self):
        """Can create a basic result with required fields."""
        result = TranscriptionResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.segments == []
        assert result.language is None
        assert result.duration is None
        assert result.provider == ""

    def test_creation_with_segments(self):
        """Can create result with segments."""
        segments = [
            TranscriptionSegment(0.0, 1.5, "Hello"),
            TranscriptionSegment(2.0, 3.5, "World"),
        ]
        result = TranscriptionResult(
            text="Hello World",
            segments=segments,
            language="en",
            duration=4.0,
            provider="whisper-local",
        )
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.duration == 4.0
        assert result.provider == "whisper-local"

    def test_to_dict(self):
        """to_dict() returns all fields with serialized segments."""
        result = TranscriptionResult(
            text="Test text",
            segments=[TranscriptionSegment(0.0, 1.0, "Test")],
            language="en",
            duration=1.0,
            provider="test",
        )
        d = result.to_dict()
        assert d["text"] == "Test text"
        assert d["language"] == "en"
        assert d["duration"] == 1.0
        assert d["provider"] == "test"
        assert len(d["segments"]) == 1
        assert d["segments"][0]["text"] == "Test"

    def test_from_dict(self):
        """from_dict() creates result from dictionary."""
        data = {
            "text": "From dict",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Seg 1"},
                {"start": 1.5, "end": 2.5, "text": "Seg 2"},
            ],
            "language": "es",
            "duration": 3.0,
            "provider": "openai",
        }
        result = TranscriptionResult.from_dict(data)
        assert result.text == "From dict"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Seg 1"
        assert result.segments[1].text == "Seg 2"
        assert result.language == "es"
        assert result.duration == 3.0
        assert result.provider == "openai"

    def test_from_dict_minimal(self):
        """from_dict() works with only required fields."""
        data = {"text": "Minimal"}
        result = TranscriptionResult.from_dict(data)
        assert result.text == "Minimal"
        assert result.segments == []
        assert result.language is None
        assert result.provider == ""

    def test_round_trip(self):
        """Round-trip: result -> to_dict -> from_dict -> matches original."""
        original = TranscriptionResult(
            text="Full round trip",
            segments=[
                TranscriptionSegment(0.0, 5.0, "First", confidence=0.9),
                TranscriptionSegment(5.5, 10.0, "Second", speaker="A"),
            ],
            language="fr",
            duration=10.0,
            provider="deepgram",
        )
        reconstructed = TranscriptionResult.from_dict(original.to_dict())
        assert reconstructed.text == original.text
        assert len(reconstructed.segments) == len(original.segments)
        assert reconstructed.segments[0].confidence == original.segments[0].confidence
        assert reconstructed.segments[1].speaker == original.segments[1].speaker
        assert reconstructed.language == original.language
        assert reconstructed.duration == original.duration
        assert reconstructed.provider == original.provider


class TestSRTFormat:
    """Tests for SRT format conversion."""

    def test_empty_segments(self):
        """to_srt() returns empty string for no segments."""
        result = TranscriptionResult(text="No segments")
        assert result.to_srt() == ""

    def test_single_segment(self):
        """to_srt() formats single segment correctly."""
        result = TranscriptionResult(
            text="Hello",
            segments=[TranscriptionSegment(0.0, 1.5, "Hello")],
        )
        srt = result.to_srt()
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "Hello" in srt

    def test_multiple_segments(self):
        """to_srt() formats multiple segments with incrementing indices."""
        result = TranscriptionResult(
            text="Hello World",
            segments=[
                TranscriptionSegment(0.0, 1.5, "Hello"),
                TranscriptionSegment(2.0, 3.5, "World"),
            ],
        )
        srt = result.to_srt()
        lines = srt.strip().split("\n\n")
        assert len(lines) == 2
        assert lines[0].startswith("1\n")
        assert lines[1].startswith("2\n")

    def test_srt_time_format_comma(self):
        """SRT uses comma for milliseconds separator."""
        result = TranscriptionResult(
            text="Test",
            segments=[TranscriptionSegment(1.0, 5.5, "Test")],
        )
        srt = result.to_srt()
        # SRT format: HH:MM:SS,mmm (comma, not period)
        assert "00:00:01,000 --> 00:00:05,500" in srt
        # Verify comma is used, not period
        assert "," in srt
        # Should NOT have periods in timestamps (except in text)
        timestamp_part = srt.split("Test")[0]
        assert "." not in timestamp_part

    def test_srt_time_hours(self):
        """SRT handles times over an hour."""
        result = TranscriptionResult(
            text="Long",
            segments=[TranscriptionSegment(3661.5, 3665.0, "Long")],  # 1:01:01.5
        )
        srt = result.to_srt()
        assert "01:01:01,500 --> 01:01:05,000" in srt

    def test_srt_time_zero_padding(self):
        """SRT pads all components to correct width."""
        result = TranscriptionResult(
            text="Pad",
            segments=[TranscriptionSegment(0.001, 0.999, "Pad")],
        )
        srt = result.to_srt()
        # Should have 00:00:00,001 format
        assert "00:00:00,001" in srt
        assert "00:00:00,999" in srt

    def test_valid_srt_structure(self):
        """to_srt() produces valid SRT structure."""
        result = TranscriptionResult(
            text="Valid SRT test",
            segments=[
                TranscriptionSegment(0.0, 2.0, "Line one"),
                TranscriptionSegment(3.0, 5.0, "Line two"),
            ],
        )
        srt = result.to_srt()
        # Each entry should have: index, timestamp, text, blank line
        entries = srt.strip().split("\n\n")
        for i, entry in enumerate(entries, 1):
            lines = entry.split("\n")
            assert len(lines) >= 3
            assert lines[0] == str(i)  # Index
            assert " --> " in lines[1]  # Timestamp
            assert len(lines[2]) > 0  # Text


class TestVTTFormat:
    """Tests for WebVTT format conversion."""

    def test_empty_segments(self):
        """to_vtt() returns just header for no segments."""
        result = TranscriptionResult(text="No segments")
        vtt = result.to_vtt()
        assert vtt.startswith("WEBVTT")

    def test_single_segment(self):
        """to_vtt() formats single segment correctly."""
        result = TranscriptionResult(
            text="Hello",
            segments=[TranscriptionSegment(0.0, 1.5, "Hello")],
        )
        vtt = result.to_vtt()
        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:01.500" in vtt
        assert "Hello" in vtt

    def test_vtt_time_format_period(self):
        """VTT uses period for milliseconds separator."""
        result = TranscriptionResult(
            text="Test",
            segments=[TranscriptionSegment(1.0, 5.5, "Test")],
        )
        vtt = result.to_vtt()
        # VTT format: HH:MM:SS.mmm (period, not comma)
        assert "00:00:01.000 --> 00:00:05.500" in vtt
        # Should NOT have comma in timestamps
        timestamp_section = vtt.split("WEBVTT")[1].split("Test")[0]
        assert "," not in timestamp_section

    def test_vtt_has_header(self):
        """VTT always starts with WEBVTT header."""
        result = TranscriptionResult(
            text="Test",
            segments=[TranscriptionSegment(0.0, 1.0, "Test")],
        )
        vtt = result.to_vtt()
        assert vtt.startswith("WEBVTT")

    def test_vtt_time_hours(self):
        """VTT handles times over an hour."""
        result = TranscriptionResult(
            text="Long",
            segments=[TranscriptionSegment(7325.0, 7330.0, "Long")],  # 2:02:05.0
        )
        vtt = result.to_vtt()
        assert "02:02:05.000 --> 02:02:10.000" in vtt

    def test_valid_vtt_structure(self):
        """to_vtt() produces valid WebVTT structure."""
        result = TranscriptionResult(
            text="Valid VTT test",
            segments=[
                TranscriptionSegment(0.0, 2.0, "Line one"),
                TranscriptionSegment(3.0, 5.0, "Line two"),
            ],
        )
        vtt = result.to_vtt()
        # Should start with WEBVTT
        lines = vtt.split("\n")
        assert lines[0].strip() == "WEBVTT"
        # Each cue should have timestamp and text
        assert vtt.count(" --> ") == 2


class TestTimeFormatting:
    """Tests for time formatting edge cases."""

    def test_srt_exactly_zero(self):
        """Handles exactly 0 seconds."""
        assert TranscriptionResult._format_srt_time(0.0) == "00:00:00,000"

    def test_vtt_exactly_zero(self):
        """Handles exactly 0 seconds."""
        assert TranscriptionResult._format_vtt_time(0.0) == "00:00:00.000"

    def test_srt_fractional_seconds(self):
        """Handles fractional seconds correctly."""
        assert TranscriptionResult._format_srt_time(0.5) == "00:00:00,500"
        # Use exact values that don't have floating point issues
        assert TranscriptionResult._format_srt_time(1.0) == "00:00:01,000"
        assert TranscriptionResult._format_srt_time(59.5) == "00:00:59,500"

    def test_vtt_fractional_seconds(self):
        """Handles fractional seconds correctly."""
        assert TranscriptionResult._format_vtt_time(0.5) == "00:00:00.500"
        assert TranscriptionResult._format_vtt_time(1.0) == "00:00:01.000"
        assert TranscriptionResult._format_vtt_time(59.5) == "00:00:59.500"

    def test_minute_boundary(self):
        """Handles minute boundaries."""
        assert TranscriptionResult._format_srt_time(60.0) == "00:01:00,000"
        assert TranscriptionResult._format_srt_time(119.5) == "00:01:59,500"

    def test_hour_boundary(self):
        """Handles hour boundaries."""
        assert TranscriptionResult._format_srt_time(3600.0) == "01:00:00,000"
        assert TranscriptionResult._format_vtt_time(7199.5) == "01:59:59.500"

    def test_large_time(self):
        """Handles large time values (24+ hours)."""
        # 25 hours = 90000 seconds
        srt = TranscriptionResult._format_srt_time(90000.0)
        assert srt == "25:00:00,000"


class TestPydanticModels:
    """Tests for Pydantic model functionality."""

    def test_pydantic_import_check(self):
        """Check if pydantic is available for structured output tests."""
        try:
            import pydantic  # noqa: F401

            pytest.importorskip("pydantic")
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_visual_entity_creation(self):
        """Can create VisualEntity with pydantic."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import get_visual_entity_model

        visual_entity_cls = get_visual_entity_model()
        entity = visual_entity_cls(
            name="Python logo",
            category="object",
            first_seen_sec=12.5,
            confidence=0.9,
            attributes={"color": "blue"},
        )
        assert entity.name == "Python logo"
        assert entity.category == "object"
        assert entity.first_seen_sec == 12.5

    def test_visual_entity_json_schema(self):
        """VisualEntity can generate JSON Schema."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import get_visual_entity_model

        visual_entity_cls = get_visual_entity_model()
        schema = visual_entity_cls.model_json_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "category" in schema["properties"]

    def test_semantic_concept_creation(self):
        """Can create SemanticConcept with pydantic."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import get_semantic_concept_model

        semantic_concept_cls = get_semantic_concept_model()
        concept = semantic_concept_cls(
            term="Machine Learning",
            definition="A subset of AI",
            importance="primary",
            first_mention_sec=45.0,
            related_terms=["AI", "deep learning"],
        )
        assert concept.term == "Machine Learning"
        assert concept.importance == "primary"

    def test_entity_extraction_result_creation(self):
        """Can create EntityExtractionResult with pydantic."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import (
            get_entity_extraction_result_model,
            get_visual_entity_model,
        )

        visual_entity_cls = get_visual_entity_model()
        entity_extraction_cls = get_entity_extraction_result_model()

        entity = visual_entity_cls(
            name="test",
            category="object",
            first_seen_sec=0.0,
        )
        result = entity_extraction_cls(objects=[entity])
        assert len(result.objects) == 1
        assert result.objects[0].name == "test"

    def test_entity_extraction_result_json_schema(self):
        """EntityExtractionResult can generate JSON Schema."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import get_entity_extraction_result_model

        entity_extraction_cls = get_entity_extraction_result_model()
        schema = entity_extraction_cls.model_json_schema()
        assert "properties" in schema
        assert "objects" in schema["properties"]
        assert "concepts" in schema["properties"]

    def test_visual_description_creation(self):
        """Can create VisualDescription with pydantic."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import get_visual_description_model

        visual_description_cls = get_visual_description_model()
        desc = visual_description_cls(
            description="A presenter at a whiteboard",
            objects=["whiteboard", "marker"],
            people=["presenter"],
            setting="Conference room",
        )
        assert desc.description == "A presenter at a whiteboard"
        assert "whiteboard" in desc.objects

    def test_visual_description_json_serializable(self):
        """VisualDescription can be serialized to JSON."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import get_visual_description_model

        visual_description_cls = get_visual_description_model()
        desc = visual_description_cls(
            description="Test scene",
            objects=["obj1"],
        )
        json_str = desc.model_dump_json()
        assert "Test scene" in json_str
        assert "obj1" in json_str


class TestLazyLoadingWrappers:
    """Tests for lazy-loading Pydantic model wrappers."""

    def test_lazy_wrapper_instantiation(self):
        """Lazy wrappers can instantiate when pydantic is available."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import VisualEntity

        entity = VisualEntity(
            name="Test",
            category="object",
            first_seen_sec=0.0,
        )
        assert entity.name == "Test"

    def test_lazy_wrapper_attribute_access(self):
        """Lazy wrappers forward attribute access to real model."""
        pytest.importorskip("pydantic")
        from claudetube.providers.types import VisualEntity

        # Should be able to access class-level attributes
        schema = VisualEntity.model_json_schema()
        assert "properties" in schema


class TestModuleExports:
    """Tests for module exports."""

    def test_transcription_types_always_exported(self):
        """TranscriptionSegment and TranscriptionResult always available."""
        from claudetube.providers.types import (
            TranscriptionResult,
            TranscriptionSegment,
        )

        assert TranscriptionSegment is not None
        assert TranscriptionResult is not None

    def test_pydantic_accessors_exported(self):
        """Pydantic model accessor functions are exported."""
        from claudetube.providers.types import (
            get_entity_extraction_result_model,
            get_semantic_concept_model,
            get_visual_description_model,
            get_visual_entity_model,
        )

        assert callable(get_visual_entity_model)
        assert callable(get_semantic_concept_model)
        assert callable(get_entity_extraction_result_model)
        assert callable(get_visual_description_model)

    def test_lazy_wrappers_exported(self):
        """Lazy-loading model wrappers are exported."""
        from claudetube.providers.types import (
            EntityExtractionResult,
            SemanticConcept,
            VisualDescription,
            VisualEntity,
        )

        assert VisualEntity is not None
        assert SemanticConcept is not None
        assert EntityExtractionResult is not None
        assert VisualDescription is not None

    def test_exports_from_package(self):
        """Types are re-exported from providers package."""
        from claudetube.providers import (
            TranscriptionResult,
            TranscriptionSegment,
        )

        assert TranscriptionSegment is not None
        assert TranscriptionResult is not None
