"""
Validators Module
=================
Input validation and clamping utilities for the Scoring & Aggregation Agent.

This module provides deterministic validation and boundary protection
for numeric inputs to ensure all values remain within valid ranges.
"""

from typing import Any, Dict


def clamp_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a numeric value to the specified range.

    Args:
        value: The value to clamp
        min_val: Minimum allowed value (default: 0.0)
        max_val: Maximum allowed value (default: 1.0)

    Returns:
        float: The clamped value within [min_val, max_val]

    Examples:
        >>> clamp_value(1.5, 0.0, 1.0)
        1.0
        >>> clamp_value(-0.2, 0.0, 1.0)
        0.0
        >>> clamp_value(0.5, 0.0, 1.0)
        0.5
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Value must be numeric, got {type(value)}")

    return max(min_val, min(max_val, float(value)))


def validate_and_clamp_dict(
    data: Dict[str, Any],
    min_val: float = 0.0,
    max_val: float = 1.0
) -> Dict[str, Any]:
    """
    Recursively validate and clamp all numeric values in a dictionary.

    This function traverses nested dictionaries and clamps all numeric
    values to the specified range. Non-numeric values are preserved.

    Args:
        data: Dictionary to validate and clamp
        min_val: Minimum allowed value (default: 0.0)
        max_val: Maximum allowed value (default: 1.0)

    Returns:
        Dict: Dictionary with all numeric values clamped

    Examples:
        >>> validate_and_clamp_dict({"score": 1.2, "name": "test"})
        {'score': 1.0, 'name': 'test'}
    """
    result = {}

    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = validate_and_clamp_dict(value, min_val, max_val)
        elif isinstance(value, (int, float)):
            # Clamp numeric values
            result[key] = clamp_value(value, min_val, max_val)
        else:
            # Preserve non-numeric values (strings, lists, etc.)
            result[key] = value

    return result


def validate_score_range(score: float, score_name: str = "score") -> None:
    """
    Validate that a score is within the valid 0-100 range.

    Args:
        score: The score to validate
        score_name: Name of the score for error messages

    Raises:
        ValueError: If score is outside valid range or invalid type
    """
    if not isinstance(score, (int, float)):
        raise ValueError(
            f"{score_name} must be numeric, got {type(score).__name__}"
        )

    if score < 0 or score > 100:
        raise ValueError(
            f"{score_name} must be between 0 and 100, got {score}"
        )


def round_score(score: float, decimals: int = 2) -> float:
    """
    Round a score to specified decimal places.

    Args:
        score: The score to round
        decimals: Number of decimal places (default: 2)

    Returns:
        float: Rounded score

    Examples:
        >>> round_score(85.6789)
        85.68
        >>> round_score(100.001)
        100.0
    """
    return round(float(score), decimals)
