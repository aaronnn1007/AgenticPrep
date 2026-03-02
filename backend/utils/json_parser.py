"""
JSON Parser with Retry Logic
=============================
Robust JSON parsing utility for LLM outputs with automatic retry mechanism.

Features:
- Handles markdown code blocks (```json)
- Automatic retry on parse failures
- Value clamping for numeric ranges
- Detailed error logging
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """Custom exception for JSON parsing failures."""
    pass


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that may contain markdown code blocks or prose.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks
    if "```json" in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

    if "```" in text:
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to find JSON object in text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return text.strip()


def clamp_value(value: Any, min_val: float = 0.0, max_val: float = 1.0) -> Any:
    """
    Clamp numeric values to specified range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value or original if not numeric
    """
    if isinstance(value, (int, float)):
        return max(min_val, min(max_val, value))
    return value


def clamp_dict_values(data: Dict[str, Any], numeric_keys: list, min_val: float = 0.0, max_val: float = 1.0) -> Dict[str, Any]:
    """
    Clamp numeric values in dictionary to specified range.

    Args:
        data: Dictionary with values to clamp
        numeric_keys: List of keys that should be clamped
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Dictionary with clamped values
    """
    clamped_data = data.copy()
    for key in numeric_keys:
        if key in clamped_data:
            original_value = clamped_data[key]
            clamped_data[key] = clamp_value(original_value, min_val, max_val)
            if original_value != clamped_data[key]:
                logger.warning(
                    f"Clamped '{key}' from {original_value} to {clamped_data[key]} "
                    f"(range: {min_val}-{max_val})"
                )
    return clamped_data


def parse_llm_json_with_retry(
    llm_call: Callable[[], str],
    max_retries: int = 2,
    numeric_keys: Optional[list] = None,
    clamp_range: tuple = (0.0, 1.0)
) -> Dict[str, Any]:
    """
    Parse JSON from LLM output with automatic retry on failure.

    Args:
        llm_call: Callable that returns LLM response as string
        max_retries: Maximum number of retry attempts (default: 2)
        numeric_keys: List of keys to clamp to numeric range
        clamp_range: Tuple of (min, max) for clamping

    Returns:
        Parsed and validated dictionary

    Raises:
        JSONParseError: If parsing fails after all retries
    """
    attempts = 0
    last_error = None

    while attempts <= max_retries:
        try:
            # Get LLM response
            raw_response = llm_call()
            logger.debug(
                f"Attempt {attempts + 1}: Raw LLM response: {raw_response[:200]}...")

            # Extract JSON from response
            json_text = extract_json_from_text(raw_response)

            # Parse JSON
            parsed_data = json.loads(json_text)

            # Clamp numeric values if specified
            if numeric_keys:
                parsed_data = clamp_dict_values(
                    parsed_data,
                    numeric_keys,
                    min_val=clamp_range[0],
                    max_val=clamp_range[1]
                )

            logger.info(f"Successfully parsed JSON on attempt {attempts + 1}")
            return parsed_data

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                f"Attempt {attempts + 1} failed: JSON decode error: {e}")
            attempts += 1

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1

    # All retries exhausted
    error_msg = f"Failed to parse JSON after {max_retries + 1} attempts. Last error: {last_error}"
    logger.error(error_msg)
    raise JSONParseError(error_msg)


def parse_json_safe(
    text: str,
    numeric_keys: Optional[list] = None,
    clamp_range: tuple = (0.0, 1.0),
    fallback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely parse JSON with fallback option.

    Args:
        text: Text containing JSON
        numeric_keys: List of keys to clamp
        clamp_range: Tuple of (min, max) for clamping
        fallback: Dictionary to return if parsing fails

    Returns:
        Parsed dictionary or fallback
    """
    try:
        json_text = extract_json_from_text(text)
        parsed_data = json.loads(json_text)

        if numeric_keys:
            parsed_data = clamp_dict_values(
                parsed_data,
                numeric_keys,
                min_val=clamp_range[0],
                max_val=clamp_range[1]
            )

        return parsed_data

    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        if fallback is not None:
            logger.info("Using fallback values")
            return fallback
        raise
