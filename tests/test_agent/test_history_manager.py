"""Tests for HistoryManager size caching and trimming functionality."""

import json
from unittest.mock import patch

import pytest

from app.agent.history_manager import HistoryManager, HistoryMessage
from app.core.config import UserSettings


@pytest.fixture
def history_manager():
    """Create HistoryManager instance for testing."""
    return HistoryManager()


@pytest.fixture
def sample_user_settings():
    """Create sample user settings for testing."""
    return UserSettings(
        llm_max_context_chars=1000,
        llm_model_name="test-model",
        llm_temperature=0.7,
        llm_api_key="test-key",
    )


class TestSizeCalculation:
    """Test size calculation functionality."""

    def test_calculate_content_size_string(self, history_manager):
        """Test size calculation for string content."""
        content = "Hello world"
        size = history_manager._calculate_content_size(content)
        assert size == len(content)

    def test_calculate_content_size_dict(self, history_manager):
        """Test size calculation for dict content matches JSON serialization."""
        content = {"thought": "test", "answer": "response"}
        size = history_manager._calculate_content_size(content)
        expected = len(json.dumps(content))
        assert size == expected

    def test_calculate_content_size_none(self, history_manager):
        """Test size calculation for None content."""
        size = history_manager._calculate_content_size(None)
        assert size == 0

    def test_history_size_calculation_uses_cached_sizes(self, history_manager):
        """Test that history trimming uses cached sizes efficiently."""
        history = [
            HistoryMessage(role="user", content="test", _size=4),
            HistoryMessage(role="assistant", content={"thought": "hi", "answer": "hello"}, _size=32),
        ]
        total = sum(msg.size for msg in history)
        assert total == 36


class TestAddMethods:
    """Test that add_* methods properly cache sizes."""

    def test_add_user_message_caches_size(self, history_manager):
        """Test add_user_message caches content size."""
        history = []
        content = "Hello world"
        history_manager.add_user_message(history, content)

        assert len(history) == 1
        message = history[0]
        assert message.role == "user"
        assert message.content == content
        assert message.size == len(content)

    def test_add_final_response_caches_size(self, history_manager):
        """Test add_final_response caches JSON dict size."""
        history = []
        final_response = {"thought": "Testing", "answer": "Hello"}
        history_manager.add_final_response(history, final_response)

        assert len(history) == 1
        message = history[0]
        assert message.role == "assistant"
        assert message.type == "final_response"
        assert message.content == final_response
        assert message.size == len(json.dumps(final_response))

    def test_add_tool_call_caches_size(self, history_manager):
        """Test add_tool_call caches JSON dict size."""
        history = []
        tool_call = {"thought": "Using tool", "tool_name": "test_tool", "args": {"param": "value"}}
        history_manager.add_tool_call(history, tool_call)

        assert len(history) == 1
        message = history[0]
        assert message.role == "assistant"
        assert message.type == "tool_call"
        assert message.content == tool_call
        assert message.size == len(json.dumps(tool_call))

    def test_add_tool_result_caches_size(self, history_manager):
        """Test add_tool_result caches result size."""
        history = []
        tool_name = "test_tool"
        result = '{"status": "success", "data": "test"}'
        history_manager.add_tool_result(history, tool_name, result)

        assert len(history) == 1
        message = history[0]
        assert message.role == "user"
        assert message.tool_result is True
        assert message.tool_name == tool_name
        assert message.content == result
        # Size should include the observation tag overhead
        expected_size = len(result) + len(f'<observation tool="{tool_name}">') + len("</observation>")
        assert message.size == expected_size

    def test_add_error_message_caches_size(self, history_manager):
        """Test add_error_message caches content size."""
        history = []
        error_msg = "Something went wrong"
        history_manager.add_error_message(history, error_msg)

        assert len(history) == 1
        message = history[0]
        expected_content = f"Error: {error_msg} Please provide a valid response."
        assert message.content == expected_content
        assert message.size == len(expected_content)


class TestTrimming:
    """Test history trimming functionality."""

    def test_trim_history_respects_context_limit(self, history_manager, sample_user_settings):
        """Test that trim_history respects context limit and preserves system prompt."""
        history = [
            HistoryMessage(role="system", content="You are an AI", _size=50),
        ]

        # Add messages that will exceed context limit
        for i in range(10):
            content = f"Message {i} with some content to exceed limits"
            size = len(content)
            history.append(HistoryMessage(role="user", content=content, _size=size))

        sample_user_settings.llm_max_context_chars = 200
        original_length = len(history)

        history_manager.trim_history(history, sample_user_settings)

        assert len(history) < original_length
        assert history[0].role == "system"  # System prompt preserved
        assert len(history) >= 2  # At least system + one message

    def test_trim_history_handles_missing_sizes(self, history_manager, sample_user_settings):
        """Test trimming works with messages that don't have cached sizes."""
        history = [
            HistoryMessage(role="system", content="You are an AI", _size=len("You are an AI")),
        ]

        for i in range(10):
            content = f"Message {i} with content"
            history.append(HistoryMessage(role="user", content=content, _size=len(content)))

        sample_user_settings.llm_max_context_chars = 200
        original_length = len(history)

        history_manager.trim_history(history, sample_user_settings)

        assert len(history) < original_length
        assert history[0].role == "system"

    def test_trim_history_does_not_trim_below_minimum(self, history_manager, sample_user_settings):
        """Test that trim_history preserves minimum history length."""
        history = [
            HistoryMessage(role="system", content="You are an AI", _size=50),
            HistoryMessage(role="user", content="test", _size=4),
        ]

        sample_user_settings.llm_max_context_chars = 1  # Extremely low limit

        history_manager.trim_history(history, sample_user_settings)

        assert len(history) == 2  # Minimum preserved


class TestErrorHandling:
    """Test error handling in size calculations."""

    def test_json_serialization_error_fallback(self, history_manager):
        """Test handling of objects that can't be JSON serialized."""

        class NonSerializable:
            pass

        content = {"data": NonSerializable()}
        size = history_manager._calculate_content_size(content)

        # Should fall back to string representation
        expected_fallback_size = len(str(content))
        assert size == expected_fallback_size

    def test_json_serialization_type_error_fallback(self, history_manager):
        """Test handling of TypeError during JSON serialization."""

        class NonSerializable:
            pass

        content = {"data": NonSerializable()}

        with patch("json.dumps", side_effect=TypeError("Object not serializable")):
            size = history_manager._calculate_content_size(content)
            expected_fallback_size = len(str(content))
            assert size == expected_fallback_size
