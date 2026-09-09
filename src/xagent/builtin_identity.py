"""Stable normalization for built-in catalog collision detection."""

from __future__ import annotations


def canonicalize_builtin_identity(value: object) -> str | None:
    """Normalize an identity only for collision checks, never persistence."""
    if value is None:
        return None
    normalized = "-".join(str(value).strip().casefold().split())
    return normalized or None
