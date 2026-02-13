from __future__ import annotations


def backoff_for_attempt(attempt_idx: int, backoff: list[float]) -> float:
    """
    Return backoff seconds for retry attempt index.

    attempt_idx is zero-based for retries: 0 means first retry wait.
    """
    if backoff:
        return float(backoff[min(attempt_idx, len(backoff) - 1)])
    return float(min(60, 2**attempt_idx))
