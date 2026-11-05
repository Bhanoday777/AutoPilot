"""Basic statistics helpers."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Arithmetic mean. Raises ValueError on an empty list."""
    if not values:
        raise ValueError("mean() of empty list")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Median of a list of numbers. Raises ValueError on an empty list."""
    if not values:
        raise ValueError("median() of empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid]
