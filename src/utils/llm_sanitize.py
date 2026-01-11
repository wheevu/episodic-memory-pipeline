"""
LLM Output Sanitization Utilities.

This module provides functions to safely extract and normalize values from 
LLM-generated JSON responses. LLM output is treated as UNTRUSTED INPUT because:

1. LLMs may return `null` instead of empty arrays/strings
2. LLMs may return wrong types (string instead of number, etc.)
3. LLMs may omit required fields entirely
4. LLMs may add unexpected fields or formatting

These helpers ensure that regardless of what the LLM returns, we get
schema-safe values that can be passed to Pydantic models without validation errors.

Usage:
    from src.utils import as_list, as_str, as_float, as_bool, as_dict
    
    result = json.loads(llm_response)
    
    topics = as_list(result.get("topics"))  # [] if null/missing/wrong type
    content = as_str(result.get("content"), default="No content")
    importance = as_float(result.get("importance"), default=0.5)
    is_worthy = as_bool(result.get("is_memory_worthy"), default=False)
"""
from typing import Any, Optional


def as_list(value: Any) -> list:
    """
    Safely convert a value to a list.
    
    Args:
        value: Any value from LLM response
        
    Returns:
        The value if it's a list, otherwise an empty list.
        
    Examples:
        >>> as_list(["a", "b"])
        ['a', 'b']
        >>> as_list(None)
        []
        >>> as_list("not a list")
        []
        >>> as_list({"key": "value"})
        []
    """
    if isinstance(value, list):
        return value
    return []


def as_str(value: Any, default: str = "") -> str:
    """
    Safely convert a value to a string.
    
    Args:
        value: Any value from LLM response
        default: Default string if value is None or not a string
        
    Returns:
        The value if it's a string, otherwise the default.
        Does NOT convert non-strings to strings (e.g., numbers stay as default).
        
    Examples:
        >>> as_str("hello")
        'hello'
        >>> as_str(None)
        ''
        >>> as_str(None, default="fallback")
        'fallback'
        >>> as_str(123)  # Numbers are not auto-converted
        ''
        >>> as_str("")  # Empty string is valid
        ''
    """
    if isinstance(value, str):
        return value
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to a float.
    
    Args:
        value: Any value from LLM response
        default: Default float if conversion fails
        
    Returns:
        The value as a float, or default if conversion fails.
        
    Examples:
        >>> as_float(0.8)
        0.8
        >>> as_float("0.75")
        0.75
        >>> as_float(None)
        0.0
        >>> as_float("invalid")
        0.0
        >>> as_float(None, default=0.5)
        0.5
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    """
    Safely convert a value to a boolean.
    
    Args:
        value: Any value from LLM response
        default: Default boolean if value cannot be converted
        
    Returns:
        The value as a boolean, or default if conversion fails.
        
    Handles:
        - Actual booleans: True, False
        - String booleans: "true", "false", "True", "False"
        - Numeric: 1, 0 (treated as True, False)
        
    Examples:
        >>> as_bool(True)
        True
        >>> as_bool("true")
        True
        >>> as_bool("false")
        False
        >>> as_bool(None)
        False
        >>> as_bool(1)
        True
        >>> as_bool("invalid")
        False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def as_dict(value: Any) -> dict:
    """
    Safely convert a value to a dictionary.
    
    Args:
        value: Any value from LLM response
        
    Returns:
        The value if it's a dict, otherwise an empty dict.
        
    Examples:
        >>> as_dict({"key": "value"})
        {'key': 'value'}
        >>> as_dict(None)
        {}
        >>> as_dict(["not", "a", "dict"])
        {}
    """
    if isinstance(value, dict):
        return value
    return {}


def sanitize_llm_response(
    response: dict,
    schema: dict[str, tuple[type, Any]]
) -> dict:
    """
    Sanitize an entire LLM response dictionary according to a schema.
    
    Args:
        response: The parsed JSON response from the LLM
        schema: A dictionary mapping field names to (type, default) tuples
                Supported types: list, str, float, bool, dict
                
    Returns:
        A sanitized dictionary with all fields conforming to their expected types.
        
    Example:
        >>> schema = {
        ...     "topics": (list, []),
        ...     "content": (str, ""),
        ...     "importance": (float, 0.5),
        ...     "is_worthy": (bool, False),
        ... }
        >>> sanitize_llm_response({"topics": None, "importance": "0.8"}, schema)
        {'topics': [], 'content': '', 'importance': 0.8, 'is_worthy': False}
    """
    if not isinstance(response, dict):
        response = {}
    
    result = {}
    
    type_handlers = {
        list: as_list,
        str: lambda v, d: as_str(v, d),
        float: lambda v, d: as_float(v, d),
        bool: lambda v, d: as_bool(v, d),
        dict: lambda v, d: as_dict(v),
    }
    
    for field, (field_type, default) in schema.items():
        value = response.get(field)
        handler = type_handlers.get(field_type)
        
        if handler:
            if field_type in (str, float, bool):
                result[field] = handler(value, default)
            else:
                result[field] = handler(value)
        else:
            # Unknown type, just use default if None
            result[field] = value if value is not None else default
    
    return result


def safe_get_nested(data: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely traverse nested dictionary keys.
    
    Args:
        data: The dictionary to traverse
        *keys: Variable number of keys to traverse
        default: Default value if any key is missing
        
    Returns:
        The value at the nested path, or default if not found.
        
    Example:
        >>> data = {"a": {"b": {"c": 1}}}
        >>> safe_get_nested(data, "a", "b", "c")
        1
        >>> safe_get_nested(data, "a", "x", "y", default="missing")
        'missing'
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def sanitize_string_list(
    items: list,
    max_items: int = 20,
    max_length: int = 100,
    strip_chars: str = " \t\n\r\"'",
) -> list[str]:
    """
    Sanitize a list of strings (topics, entities, etc.) from LLM output.
    
    Args:
        items: List of strings from LLM response
        max_items: Maximum number of items to keep
        max_length: Maximum length per item (chars)
        strip_chars: Characters to strip from each item
        
    Returns:
        A sanitized list of strings, filtered and truncated.
        
    Examples:
        >>> sanitize_string_list(["topic1", "topic2", "  topic3  "])
        ['topic1', 'topic2', 'topic3']
        >>> sanitize_string_list(["a" * 200], max_length=10)
        ['aaaaaaaaaa']
        >>> sanitize_string_list(list(range(30)), max_items=5)
        []
    """
    if not isinstance(items, list):
        return []
    
    result = []
    for item in items:
        # Only keep strings
        if not isinstance(item, str):
            continue
        
        # Strip and filter empty
        cleaned = item.strip(strip_chars)
        if not cleaned:
            continue
        
        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        result.append(cleaned)
        
        # Stop if we've reached the limit
        if len(result) >= max_items:
            break
    
    return result


def sanitize_topics(topics: Any, max_topics: int = 20, max_length: int = 100) -> list[str]:
    """
    Sanitize topics from LLM output.
    
    Args:
        topics: Topics value from LLM response
        max_topics: Maximum number of topics to keep
        max_length: Maximum length per topic (chars)
        
    Returns:
        A sanitized list of topic strings.
        
    Examples:
        >>> sanitize_topics(["work", "learning", "  health  "])
        ['work', 'learning', 'health']
        >>> sanitize_topics(None)
        []
        >>> sanitize_topics([1, 2, "valid"])
        ['valid']
    """
    items = as_list(topics)
    return sanitize_string_list(items, max_items=max_topics, max_length=max_length)


def sanitize_entities(entities: Any, max_entities: int = 50, max_length: int = 100) -> list[str]:
    """
    Sanitize entities from LLM output.
    
    Args:
        entities: Entities value from LLM response
        max_entities: Maximum number of entities to keep
        max_length: Maximum length per entity (chars)
        
    Returns:
        A sanitized list of entity strings.
        
    Examples:
        >>> sanitize_entities(["John", "OpenAI", "  Python  "])
        ['John', 'OpenAI', 'Python']
        >>> sanitize_entities(None)
        []
        >>> sanitize_entities(["a" * 200], max_length=50)
        ['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa']
    """
    items = as_list(entities)
    return sanitize_string_list(items, max_items=max_entities, max_length=max_length)

