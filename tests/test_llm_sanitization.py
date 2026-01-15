"""
Tests for LLM output sanitization utilities.

These tests verify that the sanitization layer correctly handles:
- null values from LLM responses
- wrong types (e.g., string instead of number)
- missing fields
- malformed JSON structures

The tests ensure that Pydantic models receive schema-safe values regardless
of what the LLM actually returns.
"""

from datetime import datetime

import pytest

# Import models to test integration
from src.models import Episode, Fact, FactCategory, MemoryType, Summary

# Import sanitization utilities
from src.utils.llm_sanitize import (
    as_bool,
    as_dict,
    as_float,
    as_list,
    as_str,
    safe_get_nested,
    sanitize_llm_response,
)


class TestAsListFunction:
    """Test as_list sanitization helper."""

    def test_valid_list_passthrough(self) -> None:
        """Valid lists should pass through unchanged."""
        assert as_list(["a", "b", "c"]) == ["a", "b", "c"]
        assert as_list([1, 2, 3]) == [1, 2, 3]
        assert as_list([]) == []

    def test_null_returns_empty_list(self) -> None:
        """null/None should return empty list."""
        assert as_list(None) == []

    def test_non_list_returns_empty_list(self) -> None:
        """Non-list types should return empty list."""
        assert as_list("not a list") == []
        assert as_list(123) == []
        assert as_list({"key": "value"}) == []
        assert as_list(True) == []
        assert as_list(3.14) == []

    def test_nested_list_passthrough(self) -> None:
        """Nested lists should pass through."""
        nested = [["a", "b"], ["c", "d"]]
        assert as_list(nested) == nested


class TestAsStrFunction:
    """Test as_str sanitization helper."""

    def test_valid_string_passthrough(self) -> None:
        """Valid strings should pass through unchanged."""
        assert as_str("hello") == "hello"
        assert as_str("") == ""
        assert as_str("  spaced  ") == "  spaced  "

    def test_null_returns_default(self) -> None:
        """null/None should return default."""
        assert as_str(None) == ""
        assert as_str(None, default="fallback") == "fallback"

    def test_non_string_returns_default(self) -> None:
        """Non-string types should return default, not auto-convert."""
        assert as_str(123) == ""
        assert as_str(123, default="num") == "num"
        assert as_str(True) == ""
        assert as_str(["list"]) == ""
        assert as_str({"dict": "value"}) == ""

    def test_custom_default(self) -> None:
        """Custom defaults should be used correctly."""
        assert as_str(None, default="custom") == "custom"
        assert as_str(123, default="not_a_number") == "not_a_number"


class TestAsFloatFunction:
    """Test as_float sanitization helper."""

    def test_valid_float_passthrough(self) -> None:
        """Valid floats should pass through."""
        assert as_float(3.14) == 3.14
        assert as_float(0.0) == 0.0
        assert as_float(-1.5) == -1.5

    def test_integer_converts_to_float(self) -> None:
        """Integers should convert to float."""
        assert as_float(5) == 5.0
        assert as_float(0) == 0.0

    def test_string_number_converts(self) -> None:
        """String numbers should convert to float."""
        assert as_float("3.14") == 3.14
        assert as_float("5") == 5.0
        assert as_float("-2.5") == -2.5

    def test_null_returns_default(self) -> None:
        """null/None should return default."""
        assert as_float(None) == 0.0
        assert as_float(None, default=0.5) == 0.5

    def test_invalid_string_returns_default(self) -> None:
        """Invalid strings should return default."""
        assert as_float("not a number") == 0.0
        assert as_float("invalid", default=0.5) == 0.5

    def test_complex_types_return_default(self) -> None:
        """Complex types should return default."""
        assert as_float(["list"]) == 0.0
        assert as_float({"dict": "value"}) == 0.0


class TestAsBoolFunction:
    """Test as_bool sanitization helper."""

    def test_valid_bool_passthrough(self) -> None:
        """Valid booleans should pass through."""
        assert as_bool(True) is True
        assert as_bool(False) is False

    def test_string_true_converts(self) -> None:
        """String 'true' variants should convert."""
        assert as_bool("true") is True
        assert as_bool("True") is True
        assert as_bool("TRUE") is True

    def test_string_false_converts(self) -> None:
        """String 'false' variants should convert."""
        assert as_bool("false") is False
        assert as_bool("False") is False
        assert as_bool("FALSE") is False

    def test_numeric_converts(self) -> None:
        """Numeric values should convert (0=False, non-zero=True)."""
        assert as_bool(1) is True
        assert as_bool(0) is False
        assert as_bool(-1) is True

    def test_null_returns_default(self) -> None:
        """null/None should return default."""
        assert as_bool(None) is False
        assert as_bool(None, default=True) is True

    def test_invalid_string_returns_default(self) -> None:
        """Invalid strings should return default."""
        assert as_bool("yes") is False
        assert as_bool("no") is False
        assert as_bool("invalid") is False
        assert as_bool("invalid", default=True) is True

    def test_complex_types_return_default(self) -> None:
        """Complex types should return default."""
        assert as_bool(["list"]) is False
        assert as_bool({"dict": "value"}) is False


class TestAsDictFunction:
    """Test as_dict sanitization helper."""

    def test_valid_dict_passthrough(self) -> None:
        """Valid dicts should pass through."""
        assert as_dict({"key": "value"}) == {"key": "value"}
        assert as_dict({}) == {}

    def test_null_returns_empty_dict(self) -> None:
        """null/None should return empty dict."""
        assert as_dict(None) == {}

    def test_non_dict_returns_empty_dict(self) -> None:
        """Non-dict types should return empty dict."""
        assert as_dict("string") == {}
        assert as_dict(123) == {}
        assert as_dict(["list"]) == {}
        assert as_dict(True) == {}


class TestSanitizeLlmResponse:
    """Test the comprehensive sanitize_llm_response function."""

    def test_full_sanitization(self) -> None:
        """Test sanitization with mixed valid and invalid values."""
        schema = {
            "topics": (list, []),
            "content": (str, ""),
            "importance": (float, 0.5),
            "is_worthy": (bool, False),
        }

        # Malformed LLM response with nulls
        response = {
            "topics": None,
            "content": None,
            "importance": "0.8",  # String instead of float
            "is_worthy": "true",  # String instead of bool
        }

        result = sanitize_llm_response(response, schema)

        assert result["topics"] == []
        assert result["content"] == ""
        assert result["importance"] == 0.8
        assert result["is_worthy"] is True

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields should use schema defaults."""
        schema = {
            "topics": (list, []),
            "content": (str, "default content"),
        }

        result = sanitize_llm_response({}, schema)

        assert result["topics"] == []
        assert result["content"] == "default content"

    def test_non_dict_response_handled(self) -> None:
        """Non-dict responses should be treated as empty."""
        schema = {
            "topics": (list, []),
            "content": (str, "fallback"),
        }

        result = sanitize_llm_response(None, schema)

        assert result["topics"] == []
        assert result["content"] == "fallback"


class TestSafeGetNested:
    """Test safe_get_nested for traversing nested dicts."""

    def test_valid_path(self) -> None:
        """Valid nested paths should return correct value."""
        data = {"a": {"b": {"c": "value"}}}
        assert safe_get_nested(data, "a", "b", "c") == "value"

    def test_missing_key_returns_default(self) -> None:
        """Missing keys should return default."""
        data = {"a": {"b": "value"}}
        assert safe_get_nested(data, "a", "x", "y") is None
        assert safe_get_nested(data, "a", "x", "y", default="missing") == "missing"

    def test_non_dict_intermediate_returns_default(self) -> None:
        """Non-dict intermediate values should return default."""
        data = {"a": "not_a_dict"}
        assert safe_get_nested(data, "a", "b") is None


class TestEpisodeModelIntegration:
    """Test that Episode model instantiates correctly with sanitized values."""

    def test_episode_with_all_nulls_sanitized(self) -> None:
        """Episode should create successfully with sanitized null values."""
        # Simulate LLM response with nulls
        llm_response = {
            "content": None,
            "memory_type": None,
            "topics": None,
            "entities": None,
            "importance": None,
        }

        # Sanitize
        content = as_str(llm_response.get("content"), default="fallback") or "fallback"
        topics = as_list(llm_response.get("topics"))
        entities = as_list(llm_response.get("entities"))
        importance = as_float(llm_response.get("importance"), default=0.5)

        memory_type_str = as_str(llm_response.get("memory_type"), default="episodic")
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            memory_type = MemoryType.EPISODIC

        # Create Episode - should not raise
        episode = Episode(
            raw_input="test input",
            content=content,
            memory_type=memory_type,
            topics=topics,
            entities=entities,
            importance=importance,
            confidence=1.0,
            occurred_at=datetime.utcnow(),
            source="test",
        )

        assert episode.content == "fallback"
        assert episode.topics == []
        assert episode.entities == []
        assert episode.importance == 0.5
        assert episode.memory_type == MemoryType.EPISODIC

    def test_episode_with_wrong_types_sanitized(self) -> None:
        """Episode should handle wrong types gracefully."""
        # Simulate LLM response with wrong types
        llm_response = {
            "content": 123,  # Should be string
            "topics": "single_topic",  # Should be list
            "entities": {"not": "a list"},  # Should be list
            "importance": "high",  # Should be float
        }

        # Sanitize
        content = as_str(llm_response.get("content"), default="fallback") or "fallback"
        topics = as_list(llm_response.get("topics"))
        entities = as_list(llm_response.get("entities"))
        importance = as_float(llm_response.get("importance"), default=0.5)

        # Create Episode - should not raise
        episode = Episode(
            raw_input="test input",
            content=content,
            memory_type=MemoryType.EPISODIC,
            topics=topics,
            entities=entities,
            importance=importance,
            confidence=1.0,
            occurred_at=datetime.utcnow(),
            source="test",
        )

        assert episode.content == "fallback"  # Number became fallback
        assert episode.topics == []  # String became empty list
        assert episode.entities == []  # Dict became empty list
        assert episode.importance == 0.5  # Invalid string became default


class TestFactModelIntegration:
    """Test that Fact model instantiates correctly with sanitized values."""

    def test_fact_with_nulls_sanitized(self) -> None:
        """Fact should create successfully with sanitized null values."""
        llm_response = {
            "content": None,
            "category": None,
            "confidence": None,
        }

        content = as_str(llm_response.get("content"), default="Unknown fact")
        category_str = as_str(llm_response.get("category"), default="personal")
        confidence = as_float(llm_response.get("confidence"), default=0.8)

        try:
            category = FactCategory(category_str)
        except ValueError:
            category = FactCategory.PERSONAL

        fact = Fact(
            content=content or "Unknown fact",
            category=category,
            topic="test",
            confidence=confidence,
        )

        assert fact.content == "Unknown fact"
        assert fact.category == FactCategory.PERSONAL
        assert fact.confidence == 0.8


class TestSummaryModelIntegration:
    """Test that Summary model instantiates correctly with sanitized values."""

    def test_summary_with_nulls_sanitized(self) -> None:
        """Summary should create successfully with sanitized null values."""
        llm_response = {
            "summary": None,
            "key_events": None,
            "themes": None,
            "notable_changes": None,
        }

        content = (
            as_str(llm_response.get("summary"), default="Summary unavailable")
            or "Summary unavailable"
        )
        key_events = as_list(llm_response.get("key_events"))

        summary = Summary(
            content=content,
            topic="test",
            time_start=datetime.utcnow(),
            time_end=datetime.utcnow(),
            episode_count=5,
            key_events=key_events,
            summary_level=1,
        )

        assert summary.content == "Summary unavailable"
        assert summary.key_events == []


class TestMalformedLlmResponses:
    """Test handling of various malformed LLM response patterns."""

    def test_completely_empty_response(self) -> None:
        """Empty dict response should use all defaults."""
        schema = {
            "topics": (list, []),
            "content": (str, "empty"),
            "importance": (float, 0.5),
        }

        result = sanitize_llm_response({}, schema)

        assert result["topics"] == []
        assert result["content"] == "empty"
        assert result["importance"] == 0.5

    def test_extra_fields_ignored(self) -> None:
        """Extra fields not in schema should be ignored."""
        schema = {
            "topics": (list, []),
        }

        response = {
            "topics": ["a", "b"],
            "extra_field": "should be ignored",
            "another_extra": 123,
        }

        result = sanitize_llm_response(response, schema)

        assert result == {"topics": ["a", "b"]}
        assert "extra_field" not in result

    def test_nested_nulls_in_lists(self) -> None:
        """Lists containing nulls should pass through (null handling is per-item)."""
        # as_list doesn't filter contents, just ensures it's a list
        assert as_list([None, "valid", None]) == [None, "valid", None]
        assert as_list(["a", None, "b"]) == ["a", None, "b"]

    def test_deeply_nested_dict_as_value(self) -> None:
        """Dict value should be returned as empty dict for non-dict inputs."""
        assert as_dict({"nested": {"deeply": {"value": 123}}}) == {
            "nested": {"deeply": {"value": 123}}
        }


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_strings(self) -> None:
        """Unicode strings should pass through correctly."""
        assert as_str("한국어") == "한국어"
        assert as_str("日本語") == "日本語"
        assert as_str("emoji 🎉") == "emoji 🎉"

    def test_very_long_strings(self) -> None:
        """Very long strings should pass through."""
        long_string = "x" * 10000
        assert as_str(long_string) == long_string

    def test_very_large_numbers(self) -> None:
        """Very large numbers should convert correctly."""
        assert as_float(1e10) == 1e10
        assert as_float("1e10") == 1e10

    def test_negative_numbers(self) -> None:
        """Negative numbers should convert correctly."""
        assert as_float(-0.5) == -0.5
        assert as_float("-0.5") == -0.5

    def test_float_precision(self) -> None:
        """Float precision should be maintained."""
        assert as_float(0.123456789) == 0.123456789

    def test_whitespace_strings(self) -> None:
        """Whitespace strings should pass through unchanged."""
        assert as_str("   ") == "   "
        assert as_str("\t\n") == "\t\n"

    def test_empty_containers(self) -> None:
        """Empty containers should be valid."""
        assert as_list([]) == []
        assert as_dict({}) == {}
        assert as_str("") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
