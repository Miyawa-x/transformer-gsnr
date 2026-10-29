"""Validation helpers for ordered, named gradient streams."""

import re
from collections.abc import Iterable


LEGACY_STREAM_NAMES = ("tog", "tcg", "ncg")
RESERVED_STREAM_NAMES = frozenset({"outside"})
_STREAM_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_stream_names(stream_names: Iterable[str]) -> tuple[str, ...]:
    """Return a validated stream tuple without changing its update order."""

    names = tuple(stream_names)
    if not names:
        raise ValueError("At least one gradient stream is required")
    for name in names:
        if not isinstance(name, str) or not _STREAM_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Gradient stream names must match [A-Za-z][A-Za-z0-9_]*, "
                f"got {name!r}"
            )
        if name in RESERVED_STREAM_NAMES:
            raise ValueError(f"Gradient stream name {name!r} is reserved")
    if len(set(names)) != len(names):
        raise ValueError(f"Gradient stream names must be unique, got {names}")
    return names


def require_exact_streams(
    actual_names: Iterable[str],
    expected_names: tuple[str, ...],
    *,
    field: str,
) -> None:
    """Require the same stream set while allowing caller mapping reordering."""

    actual = tuple(actual_names)
    if len(actual) != len(expected_names) or set(actual) != set(expected_names):
        raise ValueError(
            f"{field} must contain exactly {expected_names}, got {actual}"
        )
