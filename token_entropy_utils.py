"""Utilities for reading entropy scalars transported by patched vLLM."""

from __future__ import annotations

import math
import statistics
from typing import Any


ENTROPY_NEGATIVE_TOLERANCE = 1e-3


def _transported_value(entry: Any) -> float:
    return float(getattr(entry, "logprob", entry))


def output_token_entropies(output: Any) -> list[float]:
    """Read one full-vocabulary entropy scalar per generated token."""
    if not output.outputs:
        return []
    token_entries = getattr(output.outputs[0], "logprobs", None)
    if token_entries is None:
        return []

    entropies = []
    for token_entry in token_entries:
        if token_entry is None:
            continue
        values = list(token_entry.values() if isinstance(token_entry, dict) else token_entry)
        if len(values) != 1:
            raise RuntimeError(
                "Entropy-enabled vLLM must return exactly one scalar per token; "
                f"received {len(values)} entries."
            )
        entropy = _transported_value(values[0])
        if (
            not math.isfinite(entropy)
            or entropy < -ENTROPY_NEGATIVE_TOLERANCE
        ):
            raise RuntimeError(
                f"Received invalid token entropy {entropy} while entropy tracking "
                "is enabled. The entropy sampler patch is likely inactive."
            )
        entropies.append(max(0.0, entropy))
    return entropies


def summarize_token_entropy(
    outputs: list[Any],
    *,
    require: bool = False,
) -> dict[str, Any]:
    response_means = []
    entropy_sum = 0.0
    token_count = 0
    for output in outputs:
        entropies = output_token_entropies(output)
        if not entropies:
            continue
        response_means.append(statistics.fmean(entropies))
        entropy_sum += math.fsum(entropies)
        token_count += len(entropies)

    if require and token_count == 0:
        raise RuntimeError(
            "Token entropy tracking was requested, but vLLM returned no entropy "
            "values. Use the dedicated entropy-enabled vLLM environment."
        )
    return {
        "token_entropy_sum": entropy_sum,
        "token_entropy_count": token_count,
        "avg_token_entropy": entropy_sum / token_count if token_count else 0.0,
        "avg_response_entropy": (
            statistics.fmean(response_means) if response_means else 0.0
        ),
        "std_response_entropy": (
            statistics.pstdev(response_means) if response_means else 0.0
        ),
    }
