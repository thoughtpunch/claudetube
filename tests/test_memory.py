"""Tests for video memory management."""

from claudetube.cache import (
    CacheManager,
    Observation,
    QAPair,
    VideoMemory,
    get_memory_dir,
    has_memory,
)


class TestObservation:
    """Tests for Observation dataclass."""

    def test_create_observation(self):
        obs = Observation(
            scene_id=5,
            type="bug_identified",
            content="Off-by-one error at line 42",
            timestamp="2024-01-01T12:00:00",
        )
        assert obs.scene_id == 5
        assert obs.type == "bug_identified"
        assert obs.content == "Off-by-one error at line 42"
        assert obs.timestamp == "2024-01-01T12:00:00"

    def test_to_dict(self):
        obs = Observation(
            scene_id=3,
            type="code_explanation",
            content="This function handles authentication",
            timestamp="2024-01-02T09:30:00",
        )
        d = obs.to_dict()
        assert d["scene_id"] == 3
        assert d["type"] == "code_explanation"
        assert d["content"] == "This function handles authentication"
        assert d["timestamp"] == "2024-01-02T09:30:00"

    def test_from_dict(self):
        d = {
            "scene_id": 7,
            "type": "person_identified",
            "content": "Speaker is John Doe",
            "timestamp": "2024-01-03T15:00:00",
        }
        obs = Observation.from_dict(d)
        assert obs.scene_id == 7
        assert obs.type == "person_identified"
        assert obs.content == "Speaker is John Doe"
        assert obs.timestamp == "2024-01-03T15:00:00"

    def test_from_dict_missing_timestamp(self):
        d = {
            "scene_id": 1,
            "type": "general",
            "content": "Some content",
        }
        obs = Observation.from_dict(d)
        assert obs.timestamp == ""

    def test_default_timestamp(self):
        obs = Observation(scene_id=0, type="test", content="test content")
        # Default timestamp should be set
        assert obs.timestamp != ""


class TestQAPair:
    """Tests for QAPair dataclass."""

    def test_create_qa_pair(self):
        qa = QAPair(
            question="What bug was fixed?",
            answer="An off-by-one error",
            relevant_scenes=[5, 8, 12],
            timestamp="2024-01-01T12:00:00",
        )
        assert qa.question == "What bug was fixed?"
        assert qa.answer == "An off-by-one error"
        assert qa.relevant_scenes == [5, 8, 12]
        assert qa.timestamp == "2024-01-01T12:00:00"

    def test_to_dict(self):
        qa = QAPair(
            question="How does auth work?",
            answer="It uses JWT tokens",
            relevant_scenes=[1, 2],
            timestamp="2024-01-02T10:00:00",
        )
        d = qa.to_dict()
        assert d["question"] == "How does auth work?"
        assert d["answer"] == "It uses JWT tokens"
        assert d["scenes"] == [1, 2]
        assert d["timestamp"] == "2024-01-02T10:00:00"

    def test_from_dict(self):
        d = {
            "question": "What library is used?",
            "answer": "React",
            "scenes": [3, 4, 5],
            "timestamp": "2024-01-03T14:00:00",
        }
        qa = QAPair.from_dict(d)
        assert qa.question == "What library is used?"
        assert qa.answer == "React"
        assert qa.relevant_scenes == [3, 4, 5]
        assert qa.timestamp == "2024-01-03T14:00:00"

    def test_from_dict_missing_fields(self):
        d = {
            "question": "Test?",
            "answer": "Yes",
        }
        qa = QAPair.from_dict(d)
        assert qa.relevant_scenes == []
        assert qa.timestamp == ""

    def test_default_values(self):
        qa = QAPair(question="Q?", answer="A")
        assert qa.relevant_scenes == []
        assert qa.timestamp != ""


class TestVideoMemory:
    """Tests for VideoMemory class."""

    def test_init_creates_memory_dir(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        assert (cache_dir / "memory").exists()
        assert memory.video_id == "video123"

    def test_record_observation(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_observation(
            scene_id=5, obs_type="bug_found", content="Bug at line 42"
        )

        # Verify it's in memory
        obs = memory.get_observations(5)
        assert len(obs) == 1
        assert obs[0]["type"] == "bug_found"
        assert obs[0]["content"] == "Bug at line 42"

        # Verify it's persisted
        assert (cache_dir / "memory" / "observations.json").exists()

    def test_record_multiple_observations_same_scene(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_observation(scene_id=3, obs_type="code", content="First")
        memory.record_observation(scene_id=3, obs_type="bug", content="Second")

        obs = memory.get_observations(3)
        assert len(obs) == 2
        assert obs[0]["content"] == "First"
        assert obs[1]["content"] == "Second"

    def test_record_observations_different_scenes(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_observation(scene_id=1, obs_type="a", content="Scene 1")
        memory.record_observation(scene_id=2, obs_type="b", content="Scene 2")

        assert len(memory.get_observations(1)) == 1
        assert len(memory.get_observations(2)) == 1

    def test_get_observations_empty(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        assert memory.get_observations(99) == []

    def test_get_all_observations(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_observation(scene_id=1, obs_type="a", content="One")
        memory.record_observation(scene_id=5, obs_type="b", content="Five")

        all_obs = memory.get_all_observations()
        assert "1" in all_obs
        assert "5" in all_obs
        assert len(all_obs) == 2

    def test_record_qa(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_qa(
            question="What bug was fixed?",
            answer="An off-by-one error",
            scenes=[5, 8],
        )

        history = memory.get_qa_history()
        assert len(history) == 1
        assert history[0]["question"] == "What bug was fixed?"
        assert history[0]["answer"] == "An off-by-one error"
        assert history[0]["scenes"] == [5, 8]

        # Verify it's persisted
        assert (cache_dir / "memory" / "qa_history.json").exists()

    def test_record_multiple_qa(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_qa(question="Q1?", answer="A1", scenes=[1])
        memory.record_qa(question="Q2?", answer="A2", scenes=[2, 3])

        history = memory.get_qa_history()
        assert len(history) == 2

    def test_get_context_for_scene(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_observation(scene_id=5, obs_type="bug", content="Bug here")
        memory.record_observation(scene_id=5, obs_type="fix", content="Fixed it")
        memory.record_qa(question="What bug?", answer="Off-by-one", scenes=[5, 8])
        memory.record_qa(question="Unrelated?", answer="Yes", scenes=[1, 2])

        context = memory.get_context_for_scene(5)
        assert len(context["observations"]) == 2
        assert len(context["related_qa"]) == 1
        assert context["related_qa"][0]["question"] == "What bug?"

    def test_get_context_for_scene_empty(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        context = memory.get_context_for_scene(99)
        assert context["observations"] == []
        assert context["related_qa"] == []

    def test_search_qa_history(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_qa(
            question="What is React?", answer="A JavaScript library", scenes=[1]
        )
        memory.record_qa(
            question="How to use hooks?", answer="Import from react", scenes=[2]
        )
        memory.record_qa(
            question="What about Vue?", answer="Another framework", scenes=[3]
        )

        # Search by question
        results = memory.search_qa_history("React")
        assert len(results) == 2  # Matches "React" in Q and "react" in A

        # Search by answer
        results = memory.search_qa_history("framework")
        assert len(results) == 1
        assert results[0]["answer"] == "Another framework"

        # Case insensitive
        results = memory.search_qa_history("REACT")
        assert len(results) == 2

    def test_search_qa_history_empty(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        results = memory.search_qa_history("anything")
        assert results == []

    def test_clear(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        memory.record_observation(scene_id=1, obs_type="a", content="Test")
        memory.record_qa(question="Q?", answer="A", scenes=[1])

        memory.clear()

        assert memory.get_observations(1) == []
        assert memory.get_qa_history() == []
        assert not (cache_dir / "memory" / "observations.json").exists()
        assert not (cache_dir / "memory" / "qa_history.json").exists()

    def test_observation_count(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        assert memory.observation_count == 0

        memory.record_observation(scene_id=1, obs_type="a", content="One")
        memory.record_observation(scene_id=1, obs_type="b", content="Two")
        memory.record_observation(scene_id=2, obs_type="c", content="Three")

        assert memory.observation_count == 3

    def test_qa_count(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory = VideoMemory("video123", cache_dir)
        assert memory.qa_count == 0

        memory.record_qa(question="Q1?", answer="A1", scenes=[1])
        memory.record_qa(question="Q2?", answer="A2", scenes=[2])

        assert memory.qa_count == 2

    def test_persistence_across_instances(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        # First instance writes data
        memory1 = VideoMemory("video123", cache_dir)
        memory1.record_observation(scene_id=5, obs_type="test", content="Persisted")
        memory1.record_qa(question="Q?", answer="A", scenes=[5])

        # Second instance reads it
        memory2 = VideoMemory("video123", cache_dir)
        assert len(memory2.get_observations(5)) == 1
        assert memory2.get_observations(5)[0]["content"] == "Persisted"
        assert len(memory2.get_qa_history()) == 1

    def test_handles_corrupted_observations_file(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        memory_dir = cache_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "observations.json").write_text("not valid json")

        memory = VideoMemory("video123", cache_dir)
        assert memory.get_all_observations() == {}

    def test_handles_corrupted_qa_file(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        memory_dir = cache_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "qa_history.json").write_text("not valid json")

        memory = VideoMemory("video123", cache_dir)
        assert memory.get_qa_history() == []


class TestMemoryHelpers:
    """Tests for memory helper functions."""

    def test_get_memory_dir_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        memory_dir = get_memory_dir(cache_dir)
        assert memory_dir.exists()
        assert memory_dir.name == "memory"
        assert memory_dir.parent == cache_dir

    def test_has_memory_false_when_no_dir(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        assert has_memory(cache_dir) is False

    def test_has_memory_false_when_empty_dir(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        (cache_dir / "memory").mkdir()
        assert has_memory(cache_dir) is False

    def test_has_memory_true_with_observations(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        memory_dir = cache_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "observations.json").write_text("{}")
        assert has_memory(cache_dir) is True

    def test_has_memory_true_with_qa(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        memory_dir = cache_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "qa_history.json").write_text("[]")
        assert has_memory(cache_dir) is True


class TestCacheManagerMemoryMethods:
    """Tests for memory methods on CacheManager."""

    def test_get_memory_dir(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        memory_dir = manager.get_memory_dir("video123")
        assert memory_dir.exists()
        assert memory_dir == tmp_path / "video123" / "memory"

    def test_has_memory(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        assert manager.has_memory("video123") is False

        # Create memory data
        memory_dir = tmp_path / "video123" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "observations.json").write_text("{}")

        assert manager.has_memory("video123") is True

    def test_get_video_memory(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        memory = manager.get_video_memory("video123")

        assert isinstance(memory, VideoMemory)
        assert memory.video_id == "video123"
        assert (tmp_path / "video123" / "memory").exists()

    def test_get_video_memory_roundtrip(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)

        # First memory writes data
        memory1 = manager.get_video_memory("video123")
        memory1.record_observation(scene_id=1, obs_type="test", content="Data")

        # Second memory reads it
        memory2 = manager.get_video_memory("video123")
        assert len(memory2.get_observations(1)) == 1
