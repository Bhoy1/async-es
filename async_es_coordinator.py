"""Bounded-staleness cohort accounting for asynchronous ES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


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


@dataclass(frozen=True)
class ComponentAsyncResult(Generic[Payload]):
    payload: Payload
    component: str
    dispatch_version: int
    completion_version: int
    staleness: int
    accepted: bool
    deferred: bool


class AlternatingComponentCoordinator(Generic[Payload]):
    """Build strict alternating component cohorts under bounded staleness.

    Results for the inactive component are retained until that component is
    active again. Their global policy age is checked both when they arrive and
    when they are promoted into the active cohort.
    """

    def __init__(
        self,
        *,
        cohort_size: int,
        max_staleness: int,
        components: tuple[str, ...] = ("B", "A"),
        initial_version: int = 0,
    ) -> None:
        if cohort_size < 1:
            raise ValueError("cohort_size must be positive")
        if max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")
        if initial_version < 0:
            raise ValueError("initial_version must be non-negative")
        if len(components) < 2 or len(set(components)) != len(components):
            raise ValueError("components must contain distinct phase names")
        self.cohort_size = int(cohort_size)
        self.max_staleness = int(max_staleness)
        self.components = tuple(components)
        self.current_version = int(initial_version)
        self.accepted_total = 0
        self.discarded_total = 0
        self.deferred_total = 0
        self._cohort: list[ComponentAsyncResult[Payload]] = []
        self._pending: dict[str, list[ComponentAsyncResult[Payload]]] = {
            component: [] for component in self.components
        }

    @property
    def active_component(self) -> str:
        return self.components[self.current_version % len(self.components)]

    @property
    def cohort_fill(self) -> int:
        return len(self._cohort)

    @property
    def pending_counts(self) -> dict[str, int]:
        return {
            component: len(results)
            for component, results in self._pending.items()
        }

    def observe(
        self,
        payload: Payload,
        *,
        component: str,
        dispatch_version: int,
    ) -> ComponentAsyncResult[Payload]:
        if component not in self.components:
            raise ValueError(f"Unknown component: {component!r}")
        staleness = self.current_version - int(dispatch_version)
        if staleness < 0:
            raise ValueError(
                "dispatch_version cannot be newer than the current policy version"
            )
        accepted = staleness <= self.max_staleness
        deferred = accepted and component != self.active_component
        result = ComponentAsyncResult(
            payload=payload,
            component=component,
            dispatch_version=int(dispatch_version),
            completion_version=self.current_version,
            staleness=staleness,
            accepted=accepted,
            deferred=deferred,
        )
        if not accepted:
            self.discarded_total += 1
        elif deferred:
            self._pending[component].append(result)
            self.accepted_total += 1
            self.deferred_total += 1
        else:
            if len(self._cohort) >= self.cohort_size:
                raise RuntimeError("active component cohort is already full")
            self._cohort.append(result)
            self.accepted_total += 1
        return result

    def cohort_ready(self) -> bool:
        return len(self._cohort) == self.cohort_size

    def commit_cohort(self) -> list[ComponentAsyncResult[Payload]]:
        if not self.cohort_ready():
            raise RuntimeError(
                f"cohort is incomplete: {len(self._cohort)}/{self.cohort_size}"
            )
        cohort = [self._at_current_version(result) for result in self._cohort]
        self._cohort = []
        self.current_version += 1
        self._promote_active_pending()
        return cohort

    def _at_current_version(
        self, result: ComponentAsyncResult[Payload]
    ) -> ComponentAsyncResult[Payload]:
        staleness = self.current_version - result.dispatch_version
        return ComponentAsyncResult(
            payload=result.payload,
            component=result.component,
            dispatch_version=result.dispatch_version,
            completion_version=self.current_version,
            staleness=staleness,
            accepted=staleness <= self.max_staleness,
            deferred=False,
        )

    def _promote_active_pending(self) -> None:
        pending = self._pending[self.active_component]
        self._pending[self.active_component] = []
        for old_result in pending:
            result = self._at_current_version(old_result)
            if not result.accepted:
                self.accepted_total -= 1
                self.discarded_total += 1
                continue
            if len(self._cohort) >= self.cohort_size:
                self._pending[self.active_component].append(result)
            else:
                self._cohort.append(result)

    def state_dict(
        self,
        serialize_payload: Callable[[Payload], Any] | None = None,
    ) -> dict[str, Any]:
        serialize = serialize_payload or (lambda payload: payload)

        def encode(result: ComponentAsyncResult[Payload]) -> dict[str, Any]:
            return {
                "payload": serialize(result.payload),
                "component": result.component,
                "dispatch_version": result.dispatch_version,
                "completion_version": result.completion_version,
                "staleness": result.staleness,
            }

        return {
            "cohort_size": self.cohort_size,
            "max_staleness": self.max_staleness,
            "components": list(self.components),
            "current_version": self.current_version,
            "accepted_total": self.accepted_total,
            "discarded_total": self.discarded_total,
            "deferred_total": self.deferred_total,
            "cohort": [encode(result) for result in self._cohort],
            "pending": {
                component: [encode(result) for result in results]
                for component, results in self._pending.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        deserialize_payload: Callable[[Any], Payload] | None = None,
    ) -> "AlternatingComponentCoordinator[Payload]":
        deserialize = deserialize_payload or (lambda payload: payload)
        coordinator = cls(
            cohort_size=int(state["cohort_size"]),
            max_staleness=int(state["max_staleness"]),
            components=tuple(state["components"]),
            initial_version=int(state["current_version"]),
        )
        coordinator.accepted_total = int(state.get("accepted_total", 0))
        coordinator.discarded_total = int(state.get("discarded_total", 0))
        coordinator.deferred_total = int(state.get("deferred_total", 0))

        def decode(item: dict[str, Any]) -> ComponentAsyncResult[Payload]:
            dispatch_version = int(item["dispatch_version"])
            completion_version = int(item.get("completion_version", coordinator.current_version))
            staleness = coordinator.current_version - dispatch_version
            return ComponentAsyncResult(
                payload=deserialize(item["payload"]),
                component=str(item["component"]),
                dispatch_version=dispatch_version,
                completion_version=completion_version,
                staleness=staleness,
                accepted=staleness <= coordinator.max_staleness,
                deferred=str(item["component"]) != coordinator.active_component,
            )

        coordinator._cohort = [decode(item) for item in state.get("cohort", [])]
        coordinator._pending = {
            component: [
                decode(item)
                for item in state.get("pending", {}).get(component, [])
            ]
            for component in coordinator.components
        }
        if any(
            result.component != coordinator.active_component
            for result in coordinator._cohort
        ):
            raise ValueError("restored active cohort contains the wrong component")
        if len(coordinator._cohort) >= coordinator.cohort_size:
            raise ValueError("restored cohort must be smaller than cohort_size")
        return coordinator
