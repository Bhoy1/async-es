"""Bounded-staleness cohort accounting for asynchronous ES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


Payload = TypeVar("Payload")


@dataclass(frozen=True)
class AsyncResult(Generic[Payload]):
    payload: Payload
    dispatch_version: int
    completion_version: int
    staleness: int
    accepted: bool


class BoundedStalenessCoordinator(Generic[Payload]):
    """Collect fixed-size ES cohorts while enforcing a policy-age bound."""

    def __init__(
        self,
        *,
        cohort_size: int,
        max_staleness: int,
        initial_version: int = 0,
    ) -> None:
        if cohort_size < 1:
            raise ValueError("cohort_size must be positive")
        if max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")
        if initial_version < 0:
            raise ValueError("initial_version must be non-negative")
        self.cohort_size = int(cohort_size)
        self.max_staleness = int(max_staleness)
        self.current_version = int(initial_version)
        self.accepted_total = 0
        self.discarded_total = 0
        self._cohort: list[AsyncResult[Payload]] = []

    @property
    def cohort_fill(self) -> int:
        return len(self._cohort)

    def observe(
        self,
        payload: Payload,
        *,
        dispatch_version: int,
    ) -> AsyncResult[Payload]:
        staleness = self.current_version - int(dispatch_version)
        if staleness < 0:
            raise ValueError(
                "dispatch_version cannot be newer than the current policy version"
            )
        accepted = staleness <= self.max_staleness
        result = AsyncResult(
            payload=payload,
            dispatch_version=int(dispatch_version),
            completion_version=self.current_version,
            staleness=staleness,
            accepted=accepted,
        )
        if accepted:
            self._cohort.append(result)
            self.accepted_total += 1
        else:
            self.discarded_total += 1
        return result

    def cohort_ready(self) -> bool:
        return len(self._cohort) == self.cohort_size

    def commit_cohort(self) -> list[AsyncResult[Payload]]:
        if not self.cohort_ready():
            raise RuntimeError(
                f"cohort is incomplete: {len(self._cohort)}/{self.cohort_size}"
            )
        cohort = self._cohort
        self._cohort = []
        self.current_version += 1
        return cohort
