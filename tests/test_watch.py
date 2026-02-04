"""Tests for watch_video operation and examination functions."""

import json
from pathlib import Path
from unittest.mock import patch

from claudetube.operations.watch import (
    _extract_key_phrases,
    examine_scene_deep,
    examine_scene_quick,
    watch_video,
)

# --- Test fixtures ---


def _make_scene(
    scene_id: int = 0,
    start_time: float = 0.0,
    end_time: float = 30.0,
    transcript_text: str = "",
    title: str | None = None,
    visual: dict | None = None,
) -> dict:
    """Create a minimal scene dict for testing."""
    scene = {
        "scene_id": scene_id,
        "start_time": start_time,
        "end_time": end_time,
        "transcript_text": transcript_text,
        "title": title,
    }
    if visual is not None:
        scene["visual"] = visual
    return scene


# --- examine_scene_quick tests ---


class TestExamineSceneQuick:
    """Tests for quick scene examination."""

    def test_transcript_match(self):
        scene = _make_scene(
            transcript_text="We need to fix the authentication bug in the server code",
        )
        findings = examine_scene_quick(scene, "authentication bug in server")
        assert len(findings) >= 1
        assert any(f["type"] == "transcript_match" for f in findings)

    def test_no_transcript_match(self):
        scene = _make_scene(
            transcript_text="Today we will talk about cooking pasta",
        )
        findings = examine_scene_quick(scene, "How does authentication work?")
        # "work" is only 1 word overlap, needs 2
        transcript_matches = [f for f in findings if f["type"] == "transcript_match"]
        assert len(transcript_matches) == 0

    def test_visual_match(self):
        scene = _make_scene(
            transcript_text="",
            visual={
                "description": "Code editor showing authentication module with error"
            },
        )
        findings = examine_scene_quick(scene, "authentication module error")
        assert any(f["type"] == "visual_match" for f in findings)

    def test_visual_string_format(self):
        """Visual data can be a plain string instead of dict."""
        scene = _make_scene(
            visual="Code editor showing authentication module",
        )
        # visual as string, not dict - should handle gracefully
        findings = examine_scene_quick(scene, "authentication code")
        # Should not crash
        assert isinstance(findings, list)

    def test_title_match(self):
        scene = _make_scene(
            title="Authentication Bug Fix",
            transcript_text="",
        )
        findings = examine_scene_quick(scene, "What is the authentication fix?")
        assert any(f["type"] == "title_match" for f in findings)

    def test_empty_scene(self):
        scene = _make_scene()
        findings = examine_scene_quick(scene, "anything")
        assert findings == []

    def test_findings_have_required_fields(self):
        scene = _make_scene(
            scene_id=3,
            start_time=45.0,
            transcript_text="The bug fix involves changing the comparison operator",
        )
        findings = examine_scene_quick(scene, "bug fix comparison")
        assert len(findings) >= 1
        for f in findings:
            assert "type" in f
            assert "description" in f
            assert "timestamp" in f
            assert "scene_id" in f
            assert "initial_confidence" in f

    def test_confidence_levels(self):
        """Transcript matches should have lower confidence than visual."""
        scene = _make_scene(
            transcript_text="authentication bug fix code",
            visual={"description": "authentication bug fix code"},
        )
        findings = examine_scene_quick(scene, "authentication bug fix code")
        transcript_findings = [f for f in findings if f["type"] == "transcript_match"]
        visual_findings = [f for f in findings if f["type"] == "visual_match"]
        if transcript_findings and visual_findings:
            assert (
                transcript_findings[0]["initial_confidence"]
                < visual_findings[0]["initial_confidence"]
            )


# --- examine_scene_deep tests ---


class TestExamineSceneDeep:
    """Tests for deep scene examination."""

    @patch("claudetube.operations.watch.extract_frames")
    def test_extracts_frames(self, mock_extract):
        mock_extract.return_value = [
            Path("/tmp/frame_001.jpg"),
            Path("/tmp/frame_002.jpg"),
        ]
        scene = _make_scene(
            scene_id=2,
            start_time=30.0,
            end_time=60.0,
            transcript_text="Fixing the auth bug by changing the operator",
        )
        findings = examine_scene_deep(
            scene,
            "What bug was fixed?",
            "test_video",
        )
        assert len(findings) >= 1
        deep_findings = [f for f in findings if f["type"] == "deep_analysis"]
        assert len(deep_findings) == 1
        assert "frame_paths" in deep_findings[0]
        assert len(deep_findings[0]["frame_paths"]) == 2

    @patch("claudetube.operations.watch.extract_frames")
    def test_includes_transcript_in_deep(self, mock_extract):
        mock_extract.return_value = []
        scene = _make_scene(
            transcript_text="The function was refactored to use async/await",
        )
        findings = examine_scene_deep(
            scene,
            "What changed?",
            "test_video",
        )
        transcript_findings = [f for f in findings if f["type"] == "deep_transcript"]
        assert len(transcript_findings) == 1

    @patch("claudetube.operations.watch.extract_frames")
    def test_handles_extraction_failure(self, mock_extract):
        mock_extract.side_effect = RuntimeError("FFmpeg not found")
        scene = _make_scene(
            transcript_text="Some content here",
        )
        findings = examine_scene_deep(
            scene,
            "question",
            "test_video",
        )
        # Should still return transcript findings, not crash
        assert isinstance(findings, list)
        deep_analysis = [f for f in findings if f["type"] == "deep_analysis"]
        assert len(deep_analysis) == 0

    @patch("claudetube.operations.watch.extract_frames")
    def test_caps_duration_at_10s(self, mock_extract):
        mock_extract.return_value = []
        scene = _make_scene(start_time=0, end_time=120)  # 2 minute scene
        examine_scene_deep(scene, "question", "test_video")
        # Check that duration passed to extract_frames is capped
        call_kwargs = mock_extract.call_args
        assert (
            call_kwargs.kwargs.get(
                "duration", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else 10
            )
            <= 10
        )


# --- watch_video tests ---


class TestWatchVideo:
    """Tests for the main watch_video orchestrator."""

    def test_video_not_cached(self, tmp_path):
        result = watch_video(
            "nonexistent_video",
            "What happens?",
            output_base=tmp_path,
        )
        assert "error" in result

    def test_no_scenes(self, tmp_path):
        # Create video dir with state.json but no scenes
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        (video_dir / "state.json").write_text(json.dumps({"video_id": "test_video"}))
        result = watch_video(
            "test_video",
            "What happens?",
            output_base=tmp_path,
        )
        assert "error" in result
        assert "scene" in result["error"].lower()

    def test_full_watch_workflow(self, tmp_path):
        """Test the complete watch workflow with mock scenes."""
        video_id = "test_watch"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()

        # Create state.json
        (video_dir / "state.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "title": "Test Video",
                }
            )
        )

        # Create scenes.json
        scenes = {
            "video_id": video_id,
            "method": "transcript",
            "scene_count": 3,
            "scenes": [
                {
                    "scene_id": 0,
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "title": "Introduction",
                    "transcript_text": "Welcome to our tutorial about fixing bugs in code",
                    "transcript": [],
                },
                {
                    "scene_id": 1,
                    "start_time": 30.0,
                    "end_time": 60.0,
                    "title": "The Bug",
                    "transcript_text": "The authentication bug was caused by an off-by-one error in the loop",
                    "transcript": [],
                },
                {
                    "scene_id": 2,
                    "start_time": 60.0,
                    "end_time": 90.0,
                    "title": "The Fix",
                    "transcript_text": "We fix the bug by changing the comparison from less than to less than or equal",
                    "transcript": [],
                },
            ],
        }
        (scenes_dir / "scenes.json").write_text(json.dumps(scenes))

        # Create enrichment dir for Q&A caching
        enrichment_dir = video_dir / "enrichment"
        enrichment_dir.mkdir()

        result = watch_video(
            video_id,
            "What bug was fixed?",
            max_iterations=5,
            output_base=tmp_path,
        )

        assert "error" not in result
        assert result["video_id"] == video_id
        assert result["question"] == "What bug was fixed?"
        assert "answer" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert "evidence" in result
        assert "examination_log" in result
        assert result["scenes_examined"] >= 1

    def test_returns_cached_qa(self, tmp_path):
        """Test that cached Q&A is returned when available."""
        video_id = "test_cached"
        video_dir = tmp_path / video_id
        video_dir.mkdir()

        # Create memory dir with cached Q&A (VideoMemory stores in memory/qa_history.json)
        memory_dir = video_dir / "memory"
        memory_dir.mkdir()
        qa_data = [
            {
                "question": "What bug was fixed?",
                "answer": "An off-by-one error in authentication",
                "timestamp": "2026-01-01T00:00:00",
            }
        ]
        (memory_dir / "qa_history.json").write_text(json.dumps(qa_data))

        # Create minimal scenes
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scene_count": 1,
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "transcript_text": "Some content",
                            "transcript": [],
                        }
                    ],
                }
            )
        )

        result = watch_video(
            video_id,
            "What bug was fixed?",
            output_base=tmp_path,
        )

        assert "error" not in result
        assert result["source"] == "cached_qa"
        assert result["scenes_examined"] == 0

    def test_max_iterations_respected(self, tmp_path):
        """Test that max_iterations limits scene examinations."""
        video_id = "test_max_iter"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()

        # Create many scenes
        scene_list = []
        for i in range(20):
            scene_list.append(
                {
                    "scene_id": i,
                    "start_time": float(i * 30),
                    "end_time": float((i + 1) * 30),
                    "title": f"Scene {i}",
                    "transcript_text": f"Content about topic {i} and some keyword stuff",
                    "transcript": [],
                }
            )

        (scenes_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scene_count": len(scene_list),
                    "scenes": scene_list,
                }
            )
        )

        result = watch_video(
            video_id,
            "keyword stuff",
            max_iterations=3,
            output_base=tmp_path,
        )

        assert "error" not in result
        assert len(result["examination_log"]) <= 3


# --- _extract_key_phrases tests ---


class TestExtractKeyPhrases:
    """Tests for key phrase extraction."""

    def test_two_word_phrases(self):
        phrases = _extract_key_phrases("fix the bug")
        assert "fix the" in phrases
        assert "the bug" in phrases

    def test_three_word_phrases(self):
        phrases = _extract_key_phrases("fix the auth bug")
        assert "fix the auth" in phrases
        assert "the auth bug" in phrases

    def test_single_word(self):
        phrases = _extract_key_phrases("hello")
        assert phrases == []

    def test_empty_string(self):
        phrases = _extract_key_phrases("")
        assert phrases == []


# --- _extract_relevant_content tests ---


class TestExtractRelevantContent:
    """Tests for relevant content extraction from transcripts."""

    def test_extracts_relevant_sentence(self):
        from claudetube.operations.watch import _extract_relevant_content

        transcript = (
            "This video is about training data. "
            "For a human to read the GPT-3 training data, "
            "it would take over 2600 years. "
            "That's an incredible amount of text."
        )
        question = "How long would it take a human to read the training data?"
        result = _extract_relevant_content(transcript, question)
        # Should find the sentence with "2600 years"
        assert "2600 years" in result

    def test_prefers_sentences_with_key_phrases(self):
        from claudetube.operations.watch import _extract_relevant_content

        transcript = (
            "Movie scripts are interesting. "
            "The training data is massive. "
            "Reading the training data would take a human over 2600 years."
        )
        question = "How long to read training data?"
        result = _extract_relevant_content(transcript, question)
        assert "2600 years" in result
        # Should not prioritize the irrelevant first sentence
        assert result.startswith("Movie scripts") is False

    def test_falls_back_to_preview_when_no_match(self):
        from claudetube.operations.watch import _extract_relevant_content

        transcript = "This is completely unrelated content about cooking recipes."
        question = "What is the authentication bug?"
        result = _extract_relevant_content(transcript, question)
        # Should fall back to first 200 chars since no keywords match
        assert result == transcript[:200]

    def test_limits_result_length(self):
        from claudetube.operations.watch import _extract_relevant_content

        # Create a long transcript with multiple relevant sentences
        transcript = (
            "The bug is in the authentication module. " * 5
            + "Fix the authentication by updating the config. " * 5
        )
        question = "authentication bug"
        result = _extract_relevant_content(transcript, question)
        # Should not exceed ~300 chars
        assert len(result) <= 350  # Allow some margin for sentence boundaries

    def test_empty_transcript(self):
        from claudetube.operations.watch import _extract_relevant_content

        result = _extract_relevant_content("", "any question")
        assert result == ""
